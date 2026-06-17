use bullet_lib::game::formats::bulletformat::ChessBoard;
use montyformat::chess::{Piece, Side};

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn flip_horizontal(mut bb: u64) -> u64 {
    bb.swap_bytes().reverse_bits()
}

fn build_bbs(pos: &ChessBoard) -> [u64; 8] {
    let mut bbs = [0u64; 8];
    for (pc, sq) in pos.into_iter() {
        let pt = 2 + usize::from(pc & 7);
        let c = usize::from(pc & 8 > 0);
        let bit = 1 << sq;
        bbs[c] |= bit;
        bbs[pt] |= bit;
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
            *bb = flip_horizontal(*bb);
        }
    }
    bbs
}

pub mod threat_inputs {
    use bullet_lib::game::{formats::bulletformat::ChessBoard, inputs};

    use montyformat::chess::{Attacks, Piece, Side};

    use super::{build_bbs, flip_view, map_bb, normalize_hm, offsets, threats::map_piece_threat};

    #[derive(Clone, Copy)]
    pub struct ThreatInputs {
        buckets: [usize; 64],
        total_features: usize,
    }

    impl ThreatInputs {
        pub const TOTAL_THREATS: usize = 2 * offsets::END;

        pub fn new(buckets: [usize; 32]) -> Self {
            let num_buckets = inputs::get_num_buckets(&buckets);

            let mut expanded = [0; 64];
            for (idx, elem) in expanded.iter_mut().enumerate() {
                *elem = buckets[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
            }

            let total_features = Self::TOTAL_THREATS + 768 * num_buckets + 768;

            Self { buckets: expanded, total_features }
        }
    }

    impl Default for ThreatInputs {
        fn default() -> Self {
            let total_features = Self::TOTAL_THREATS + 768 + 768;
            Self { buckets: [0; 64], total_features }
        }
    }

    impl inputs::SparseInputType for ThreatInputs {
        type RequiredDataType = ChessBoard;

        fn num_inputs(&self) -> usize {
            self.total_features
        }

        fn max_active(&self) -> usize {
            128 + 32
        }

        fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
            let get = |ksq| (if ksq % 8 > 3 { 7 } else { 0 }, 768 * self.buckets[usize::from(ksq)]);
            let (stm_flip, stm_bucket) = get(pos.our_ksq());
            let (ntm_flip, ntm_bucket) = get(pos.opp_ksq());

            #[rustfmt::skip]
            inputs::Chess768.map_features(pos, |stm, ntm| {
                f(
                    ThreatInputs::TOTAL_THREATS + stm ^ stm_flip,
                    ThreatInputs::TOTAL_THREATS + ntm ^ ntm_flip,
                );
                f(
                    ThreatInputs::TOTAL_THREATS + 768 + stm_bucket + (stm ^ stm_flip),
                    ThreatInputs::TOTAL_THREATS + 768 + ntm_bucket + (ntm ^ ntm_flip),
                );
            });

            let bbs = build_bbs(pos);

            let mut stm_count = 0;
            let mut stm_feats = [0; 128];
            let mut ntm_count = 0;
            let mut ntm_feats = [0; 128];
            map_threat_features(
                bbs,
                |stm| {
                    stm_feats[stm_count] = stm;
                    stm_count += 1;
                },
                |ntm| {
                    ntm_feats[ntm_count] = ntm;
                    ntm_count += 1;
                },
            );

            assert_eq!(stm_count, ntm_count);

            for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
                f(stm, ntm);
            }
        }

        fn shorthand(&self) -> String {
            todo!();
        }

        fn description(&self) -> String {
            todo!();
        }
    }

    fn map_threat_features<FStm: FnMut(usize), FNtm: FnMut(usize)>(bbs: [u64; 8], mut on_stm: FStm, mut on_ntm: FNtm) {
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
            let stm_offset = offsets::END * side;
            let ntm_offset = offsets::END * (side ^ 1);
            let opps = bbs[side ^ 1];

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
                        let enemy = (1 << dest) & opps > 0;
                        let target = pieces[dest];

                        if let Some(idx) = map_piece_threat(piece, sq ^ stm_mask, dest ^ stm_mask, target, enemy) {
                            on_stm(stm_offset + idx);
                        }

                        let ntm_target = (target + 6) % 12;
                        if let Some(idx) = map_piece_threat(piece, sq ^ ntm_mask, dest ^ ntm_mask, ntm_target, enemy) {
                            on_ntm(ntm_offset + idx);
                        }
                    });
                });
            }
        }
    }
}

pub mod pawn_pawn_inputs {
    use bullet_lib::game::{formats::bulletformat::ChessBoard, inputs};

    use montyformat::chess::{Piece, Side};

    use super::{build_bbs, flip_view, map_bb, normalize_hm, threat_inputs::ThreatInputs};

    #[derive(Clone, Copy)]
    pub struct PawnPawnInputs {
        threats: ThreatInputs,
        masks: [u64; 64],
    }

    impl PawnPawnInputs {
        pub const TOTAL_PAIRS: usize = 96 * 95 / 2;
        pub const TOTAL_THREATS: usize = ThreatInputs::TOTAL_THREATS;
        const MAX_PAIRS: usize = 16 * 15 / 2;

        pub fn new(buckets: [usize; 32], masks: [u64; 64]) -> Self {
            Self { threats: ThreatInputs::new(buckets), masks }
        }

        fn pawn_id(colour: usize, sq: usize) -> usize {
            colour * 48 + sq - 8
        }

        fn pair_index(id_a: usize, id_b: usize) -> usize {
            let lo = id_a.min(id_b);
            let hi = id_a.max(id_b);
            hi * (hi - 1) / 2 + lo
        }

        fn collect_pairs(&self, bbs: [u64; 8]) -> ([(usize, usize); Self::MAX_PAIRS], usize) {
            let friendly = bbs[Side::WHITE] & bbs[Piece::PAWN];
            let enemy = bbs[Side::BLACK] & bbs[Piece::PAWN];

            let mut pairs = [(0usize, 0usize); Self::MAX_PAIRS];
            let mut n = 0;

            self.emit_same_colour(friendly, 0, &mut pairs, &mut n);
            self.emit_cross_colour(friendly, enemy, &mut pairs, &mut n);
            self.emit_same_colour(enemy, 1, &mut pairs, &mut n);

            (pairs, n)
        }

        fn emit_same_colour(
            &self,
            bb: u64,
            colour: usize,
            pairs: &mut [(usize, usize); Self::MAX_PAIRS],
            n: &mut usize,
        ) {
            let mut outer = bb;
            while outer != 0 {
                let sq_a = outer.trailing_zeros() as usize;
                outer &= outer - 1;
                let id_a = Self::pawn_id(colour, sq_a);
                map_bb(outer & self.masks[sq_a], |sq_b| {
                    pairs[*n] = (id_a, Self::pawn_id(colour, sq_b));
                    *n += 1;
                });
            }
        }

        fn emit_cross_colour(
            &self,
            friendly: u64,
            enemy: u64,
            pairs: &mut [(usize, usize); Self::MAX_PAIRS],
            n: &mut usize,
        ) {
            map_bb(friendly, |sq_a| {
                let id_a = Self::pawn_id(0, sq_a);
                map_bb(enemy & self.masks[sq_a], |sq_b| {
                    pairs[*n] = (id_a, Self::pawn_id(1, sq_b));
                    *n += 1;
                });
            });
        }
    }

    #[allow(dead_code)]
    pub fn full_mask() -> [u64; 64] {
        [!0u64; 64]
    }

    pub fn three_file_band_mask() -> [u64; 64] {
        const A: u64 = 0x0101_0101_0101_0101;
        let mut masks = [0u64; 64];
        let mut sq = 8;
        while sq < 56 {
            let f = sq & 7;
            let mut m: u64 = A << f;
            if f > 0 {
                m |= A << (f - 1);
            }
            if f < 7 {
                m |= A << (f + 1);
            }
            masks[sq] = m;
            sq += 1;
        }
        masks
    }

    impl inputs::SparseInputType for PawnPawnInputs {
        type RequiredDataType = ChessBoard;

        fn num_inputs(&self) -> usize {
            Self::TOTAL_PAIRS + self.threats.num_inputs()
        }

        fn max_active(&self) -> usize {
            self.threats.max_active() + Self::MAX_PAIRS
        }

        fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
            self.threats.map_features(pos, |stm, ntm| {
                f(Self::TOTAL_PAIRS + stm, Self::TOTAL_PAIRS + ntm);
            });

            let bbs = build_bbs(pos);
            let stm_bbs = normalize_hm(bbs);
            let ntm_bbs = normalize_hm(flip_view(bbs));

            let (stm_pairs, stm_count) = self.collect_pairs(stm_bbs);
            let (ntm_pairs, ntm_count) = self.collect_pairs(ntm_bbs);

            assert_eq!(stm_count, ntm_count);

            for i in 0..stm_count {
                let stm_idx = Self::pair_index(stm_pairs[i].0, stm_pairs[i].1);
                let ntm_idx = Self::pair_index(ntm_pairs[i].0, ntm_pairs[i].1);
                f(stm_idx, ntm_idx);
            }
        }

        fn shorthand(&self) -> String {
            todo!();
        }

        fn description(&self) -> String {
            todo!();
        }
    }
}

mod threats {
    use montyformat::chess::Piece;

    use super::{attacks, indices, offsets};

    pub fn map_piece_threat(piece: usize, src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
        match piece {
            Piece::PAWN => map_pawn_threat(src, dest, target, enemy),
            Piece::KNIGHT => map_knight_threat(src, dest, target),
            Piece::BISHOP => map_bishop_threat(src, dest, target),
            Piece::ROOK => map_rook_threat(src, dest, target),
            Piece::QUEEN => map_queen_threat(src, dest, target),
            Piece::KING => panic!(),
            _ => unreachable!(),
        }
    }

    fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
        (table[src] & ((1 << dest) - 1)).count_ones() as usize
    }

    const fn offset_mapping<const N: usize>(a: [usize; N]) -> [usize; 12] {
        let mut res = [usize::MAX; 12];
        let mut i = 0;
        while i < N {
            res[a[i] - 2] = i;
            res[a[i] + 4] = i + N;
            i += 1;
        }
        res
    }

    fn target_is(target: usize, piece: usize) -> bool {
        target % 6 == piece - 2
    }

    fn map_pawn_threat(src: usize, dest: usize, target: usize, _enemy: bool) -> Option<usize> {
        const MAP: [usize; 12] = offset_mapping([/* Piece::PAWN, */ Piece::KNIGHT, Piece::ROOK]);
        if MAP[target] == usize::MAX
        /* || (enemy && dest > src && target_is(target, Piece::PAWN)) */
        {
            return None;
        }
        let id = if dest.abs_diff(src) == [9, 7][(dest > src) as usize] { 0 } else { 1 };
        let attack = 2 * (src % 8) + id - 1;
        Some(offsets::PAWN + MAP[target] * indices::PAWN + (src / 8 - 1) * 14 + attack)
    }

    fn map_knight_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
        const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN]);
        if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::KNIGHT) {
            return None;
        }
        let idx = indices::KNIGHT[src] + below(src, dest, &attacks::KNIGHT);
        Some(offsets::KNIGHT + MAP[target] * indices::KNIGHT[64] + idx)
    }

    fn map_bishop_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
        const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK]);
        if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::BISHOP) {
            return None;
        }
        let idx = indices::BISHOP[src] + below(src, dest, &attacks::BISHOP);
        Some(offsets::BISHOP + MAP[target] * indices::BISHOP[64] + idx)
    }

    fn map_rook_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
        const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK]);
        if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::ROOK) {
            return None;
        }
        let idx = indices::ROOK[src] + below(src, dest, &attacks::ROOK);
        Some(offsets::ROOK + MAP[target] * indices::ROOK[64] + idx)
    }

    fn map_queen_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
        const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN]);
        if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::QUEEN) {
            return None;
        }
        let idx = indices::QUEEN[src] + below(src, dest, &attacks::QUEEN);
        Some(offsets::QUEEN + MAP[target] * indices::QUEEN[64] + idx)
    }
}

mod offsets {
    use super::indices;

    pub const PAWN: usize = 0;
    pub const KNIGHT: usize = PAWN + 4 /* 6 */ * indices::PAWN;
    pub const BISHOP: usize = KNIGHT + 10 * indices::KNIGHT[64];
    pub const ROOK: usize = BISHOP + 8 * indices::BISHOP[64];
    pub const QUEEN: usize = ROOK + 8 * indices::ROOK[64];
    pub const END: usize = QUEEN + 10 * indices::QUEEN[64];
}

mod indices {
    use super::attacks;

    macro_rules! init_add_assign {
        (|$sq:ident, $init:expr, $size:literal | $($rest:tt)+) => {{
            let mut $sq = 0;
            let mut res = [{$($rest)+}; $size + 1];
            let mut val = $init;
            while $sq < $size {
                res[$sq] = val;
                val += {$($rest)+};
                $sq += 1;
            }
            res[$size] = val;
            res
        }};
    }

    pub const PAWN: usize = 84;
    pub const KNIGHT: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::KNIGHT[sq].count_ones() as usize);
    pub const BISHOP: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::BISHOP[sq].count_ones() as usize);
    pub const ROOK: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::ROOK[sq].count_ones() as usize);
    pub const QUEEN: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::QUEEN[sq].count_ones() as usize);
}

mod attacks {
    macro_rules! init {
        (|$sq:ident, $size:literal | $($rest:tt)+) => {{
            let mut $sq = 0;
            let mut res = [{$($rest)+}; $size];
            while $sq < $size {
                res[$sq] = {$($rest)+};
                $sq += 1;
            }
            res
        }};
    }

    const A: u64 = 0x0101_0101_0101_0101;
    #[allow(dead_code)]
    const H: u64 = A << 7;

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

    pub const KNIGHT: [u64; 64] = init!(|sq, 64| {
        let n = 1 << sq;
        let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
        let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
        (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
    });

    pub const BISHOP: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
    });

    pub const ROOK: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        (0xFF << (rank * 8)) ^ (A << file)
    });

    pub const QUEEN: [u64; 64] = init!(|sq, 64| BISHOP[sq] | ROOK[sq]);
}
