use std::sync::Arc;

use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{ChessBucketsMirrored, SparseInputType},
        outputs::OutputBuckets,
    },
    wdl::WdlScheduler,
};
use bullet_trainer::model::{DenseInput, ModelInputs, ModelInputsMapper, SparseInput};
use montyformat::chess::{Attacks, Piece, Side};

pub type InputTy = (((((SparseInput, SparseInput), SparseInput), SparseInput), SparseInput), DenseInput<f32>);

pub fn make_inputs_mapper(
    params: (&ModelInputs<InputTy>, &PawnPawnInputs, ChessBucketsMirrored, impl OutputBuckets<ChessBoard>),
    wdl: impl WdlScheduler,
) -> ModelInputsMapper<ChessBoard> {
    let pp = params.1.clone();

    ModelInputsMapper::build(
        params.0,
        move |pos, step, (((((stm_pp, ntm_pp), stm_psqt), ntm_psqt), bucket), target)| {
            let mut cnt = 0;
            params.2.map_features(pos, |stm, ntm| {
                stm_psqt[cnt] = stm.try_into().unwrap();
                ntm_psqt[cnt] = ntm.try_into().unwrap();
                cnt += 1;
            });

            if cnt < params.2.max_active() {
                stm_psqt[cnt] = -1;
                ntm_psqt[cnt] = -1;
            }

            let mut stm_cnt = 0;
            let mut ntm_cnt = 0;
            pp.map_features(
                pos,
                |stm| {
                    stm_pp[stm_cnt] = stm.try_into().unwrap();
                    stm_cnt += 1;
                },
                |ntm| {
                    ntm_pp[ntm_cnt] = ntm.try_into().unwrap();
                    ntm_cnt += 1;
                },
            );

            assert_eq!(stm_cnt, ntm_cnt);

            if stm_cnt < pp.max_active() {
                stm_pp[stm_cnt] = -1;
                ntm_pp[stm_cnt] = -1;
            }

            bucket[0] = i32::from(params.3.bucket(pos));

            let result = f32::from(pos.result) / 2.0;
            let score = 1.0 / (1.0 + (f32::from(-pos.score) / 400.0).exp());
            let lambda = wdl.blend(step.batch(), step.superbatch(), step.final_superbatch());
            assert!((0.0..=1.0).contains(&lambda), "WDL lambda must be in [0, 1]");
            target[0] = lambda * result + (1. - lambda) * score;
        },
    )
}

pub fn three_file_band_mask() -> [u64; 64] {
    const A: u64 = 0x0101_0101_0101_0101;
    let mut masks = [0; 64];
    for (sq, mask) in masks.iter_mut().enumerate().take(56).skip(8) {
        let f = sq & 7;
        let mut m: u64 = A << f;
        if f > 0 {
            m |= A << (f - 1);
        }
        if f < 7 {
            m |= A << (f + 1);
        }
        *mask = m;
    }
    masks
}

#[derive(Clone)]
pub struct PawnPawnInputs {
    threats: Arc<Threats>,
    masks: [u64; 64],
}

impl PawnPawnInputs {
    pub const TOTAL_PAIRS: usize = 96 * 95 / 2;
    const MAX_PAIRS: usize = 16 * 15 / 2;

    pub fn new(masks: [u64; 64]) -> Self {
        Self { threats: Arc::new(Threats::new()), masks }
    }

    pub fn _total_threats(&self) -> usize {
        self.threats.num_inputs()
    }

    pub fn num_inputs(&self) -> usize {
        Self::TOTAL_PAIRS + self.threats.num_inputs()
    }

    pub fn max_active(&self) -> usize {
        Self::MAX_PAIRS + self.threats.max_active()
    }

    fn pawn_id(colour: usize, sq: usize) -> usize {
        colour * 48 + sq - 8
    }

    fn pair_index(id_a: usize, id_b: usize) -> usize {
        let lo = id_a.min(id_b);
        let hi = id_a.max(id_b);
        hi * (hi - 1) / 2 + lo
    }

    fn emit_same_colour(&self, bb: u64, colour: usize, f: &mut impl FnMut(usize)) {
        let mut outer = bb;
        while outer != 0 {
            let sq_a = outer.trailing_zeros() as usize;
            outer &= outer - 1;
            let id_a = Self::pawn_id(colour, sq_a);
            map_bb(outer & self.masks[sq_a], |sq_b| f(Self::pair_index(id_a, Self::pawn_id(colour, sq_b))));
        }
    }

    fn collect_pairs(&self, bbs: [u64; 8], f: &mut impl FnMut(usize)) {
        let friendly = bbs[Side::WHITE] & bbs[Piece::PAWN];
        let enemy = bbs[Side::BLACK] & bbs[Piece::PAWN];

        self.emit_same_colour(friendly, 0, f);

        map_bb(friendly, |sq_a| {
            let id_a = Self::pawn_id(0, sq_a);
            map_bb(enemy & self.masks[sq_a], |sq_b| f(Self::pair_index(id_a, Self::pawn_id(1, sq_b))));
        });

        self.emit_same_colour(enemy, 1, f);
    }

    pub fn map_features(&self, pos: &ChessBoard, mut on_stm: impl FnMut(usize), mut on_ntm: impl FnMut(usize)) {
        let bbs = build_bbs(pos);
        self.threats.map(bbs, |stm| on_stm(Self::TOTAL_PAIRS + stm), |ntm| on_ntm(Self::TOTAL_PAIRS + ntm));
        self.collect_pairs(normalize_hm(bbs), &mut on_stm);
        self.collect_pairs(normalize_hm(flip_view(bbs)), &mut on_ntm);
    }
}

pub struct Threats {
    pawn_map: [usize; 12],
    non_pk_data: [PieceThreatsData; 4],
    offsets: [usize; 5],
}

impl Threats {
    pub fn new() -> Self {
        const DIAGS: [u64; 15] = [
            0x0100_0000_0000_0000,
            0x0201_0000_0000_0000,
            0x0402_0100_0000_0000,
            0x0804_0201_0000_0000,
            0x1008_0402_0100_0000,
            0x2010_0804_0201_0000,
            0x4020_1008_0402_0100,
            0x8040_2010_0804_0201,
            0x0080_4020_1008_0402,
            0x0000_8040_2010_0804,
            0x0000_0080_4020_1008,
            0x0000_0000_8040_2010,
            0x0000_0000_0080_4020,
            0x0000_0000_0000_8040,
            0x0000_0000_0000_0080,
        ];

        let knight = PieceThreatsData::new(
            Piece::KNIGHT,
            [Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN],
            |sq| {
                let n = 1 << sq;
                let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
                let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
                (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
            },
        );

        let bishop =
            PieceThreatsData::new(Piece::BISHOP, [Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK], |sq| {
                let rank = sq / 8;
                let file = sq % 8;
                DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
            });

        let rook = PieceThreatsData::new(Piece::ROOK, [Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK], |sq| {
            let rank = sq / 8;
            let file = sq % 8;
            (0xFF << (rank * 8)) ^ (0x0101_0101_0101_0101 << file)
        });

        let queen = PieceThreatsData::new(
            Piece::QUEEN,
            [Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN],
            |sq| bishop.attacks[sq] | rook.attacks[sq],
        );

        let mut offsets = [4 * 84; 5];
        for (i, &cnt) in [10 * knight.count, 8 * bishop.count, 8 * rook.count, 10 * queen.count].iter().enumerate() {
            offsets[i + 1] = offsets[i] + cnt;
        }

        Self {
            pawn_map: make_targets([Piece::KNIGHT, Piece::ROOK]),
            non_pk_data: [knight, bishop, rook, queen],
            offsets,
        }
    }

    pub fn max_active(&self) -> usize {
        128
    }

    pub fn num_inputs(&self) -> usize {
        2 * self.offsets[4]
    }

    pub fn map(&self, bbs: [u64; 8], mut on_stm: impl FnMut(usize), mut on_ntm: impl FnMut(usize)) {
        let stm_king = (bbs[Side::WHITE] & bbs[Piece::KING]).trailing_zeros() as usize;
        let ntm_king = (bbs[Side::BLACK] & bbs[Piece::KING]).trailing_zeros() as usize;
        let stm_mask = if stm_king % 8 > 3 { 7 } else { 0 };
        let ntm_mask = 56 ^ if ntm_king % 8 > 3 { 7 } else { 0 };

        let mut pieces = [13; 64];
        for side in [Side::WHITE, Side::BLACK] {
            for piece in Piece::PAWN..=Piece::KING {
                let pc = 6 * side + piece - 2;
                map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
            }
        }

        let occ = bbs[0] | bbs[1];

        for side in [Side::WHITE, Side::BLACK] {
            let stm_offset = self.offsets[4] * side;
            let ntm_offset = self.offsets[4] * (side ^ 1);

            for piece in Piece::PAWN..Piece::KING {
                map_bb(bbs[side] & bbs[piece], |sq| {
                    let threats = match piece {
                        Piece::PAWN => Attacks::pawn(sq, side),
                        Piece::KNIGHT => Attacks::knight(sq),
                        Piece::BISHOP => Attacks::bishop(sq, occ),
                        Piece::ROOK => Attacks::rook(sq, occ),
                        Piece::QUEEN => Attacks::queen(sq, occ),
                        _ => unreachable!(),
                    } & occ;

                    map_bb(threats, |dest| {
                        let target = pieces[dest];
                        if let Some(idx) = self.map_single(piece, sq ^ stm_mask, dest ^ stm_mask, target) {
                            on_stm(stm_offset + idx);
                        }

                        let ntm_target = (target + 6) % 12;
                        if let Some(idx) = self.map_single(piece, sq ^ ntm_mask, dest ^ ntm_mask, ntm_target) {
                            on_ntm(ntm_offset + idx);
                        }
                    });
                });
            }
        }
    }

    fn map_single(&self, piece: usize, src: usize, dest: usize, target: usize) -> Option<usize> {
        if piece == Piece::PAWN {
            if self.pawn_map[target] == usize::MAX {
                return None;
            }
            let id = if dest.abs_diff(src) == [9, 7][(dest > src) as usize] { 0 } else { 1 };
            let attack = 2 * (src % 8) + id - 1;
            Some(self.pawn_map[target] * 84 + (src / 8 - 1) * 14 + attack)
        } else {
            self.non_pk_data[piece - 3].map(src, dest, target, self.offsets[piece - 3])
        }
    }
}

pub struct PieceThreatsData {
    piece: usize,
    attacks: [u64; 64],
    indices: [usize; 64],
    targets: [usize; 12],
    count: usize,
}

impl PieceThreatsData {
    pub fn new<const N: usize>(piece: usize, valid: [usize; N], f: impl Fn(usize) -> u64) -> Self {
        let mut attacks = [0; 64];
        let mut indices = [0; 64];
        let mut count = 0;

        for sq in 0..64 {
            attacks[sq] = f(sq);
            indices[sq] = count;
            count += attacks[sq].count_ones() as usize;
        }

        Self { piece, attacks, indices, targets: make_targets(valid), count }
    }

    pub fn map(&self, src: usize, dest: usize, target: usize, offset: usize) -> Option<usize> {
        if self.targets[target] == usize::MAX || (dest > src && target % 6 == self.piece - 2) {
            return None;
        }
        let idx = self.indices[src] + (self.attacks[src] & ((1 << dest) - 1)).count_ones() as usize;
        Some(offset + self.targets[target] * self.count + idx)
    }
}

fn map_bb(mut bb: u64, mut f: impl FnMut(usize)) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn build_bbs(pos: &ChessBoard) -> [u64; 8] {
    let mut bbs = [0u64; 8];
    for (pc, sq) in pos.into_iter() {
        let bit = 1 << sq;
        bbs[usize::from(pc & 8 > 0)] |= bit;
        bbs[2 + usize::from(pc & 7)] |= bit;
    }
    bbs
}

fn flip_view(mut bbs: [u64; 8]) -> [u64; 8] {
    bbs.swap(Side::WHITE, Side::BLACK);
    for bb in bbs.iter_mut() {
        *bb = bb.swap_bytes();
    }
    bbs
}

fn normalize_hm(mut bbs: [u64; 8]) -> [u64; 8] {
    let ksq = (bbs[Side::WHITE] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = bb.swap_bytes().reverse_bits();
        }
    }
    bbs
}

fn make_targets<const N: usize>(valid: [usize; N]) -> [usize; 12] {
    let mut targets = [usize::MAX; 12];
    for i in 0..N {
        targets[valid[i] - 2] = i;
        targets[valid[i] + 4] = i + N;
    }
    targets
}
