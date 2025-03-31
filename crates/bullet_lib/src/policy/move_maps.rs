use crate::game::formats::montyformat::chess::{Attacks, Move, Piece, Position, Side};

pub const UNIQUE_CHESS_MOVES: usize = NUM_MOVES;

pub const MAX_MOVES: usize = 96;
const NUM_MOVES: usize = OFFSETS[64] + PROMOS;
const PROMOS: usize = 4 * 22;

#[derive(Clone, Copy)]
pub struct ChessMoveMapper<T, B> {
    pub(crate) transform: T,
    pub(crate) bucket: B,
}

#[derive(Clone, Copy)]
pub struct NoBuckets;
impl MoveBucket for NoBuckets {
    fn num_buckets(&self) -> usize {
        1
    }

    fn get(&self, _: &Position, _: Move) -> usize {
        0
    }
}

#[derive(Clone, Copy)]
pub struct GoodSEEBuckets(pub i32);
impl MoveBucket for GoodSEEBuckets {
    fn num_buckets(&self) -> usize {
        2
    }

    fn get(&self, pos: &Position, mov: Move) -> usize {
        usize::from(see(pos, &mov, self.0))
    }
}

#[derive(Clone, Copy)]
pub struct NoTransform;
impl SquareTransform for NoTransform {
    fn apply(&self, _: &Position, sq: usize) -> usize {
        sq
    }
}

#[derive(Clone, Copy)]
pub struct HorizontalMirror;
impl SquareTransform for HorizontalMirror {
    fn apply(&self, pos: &Position, sq: usize) -> usize {
        sq ^ if pos.king_index() % 8 > 3 { 7 } else { 0 }
    }
}

pub trait SquareTransform: Copy + Send + Sync + 'static {
    fn apply(&self, pos: &Position, sq: usize) -> usize;
}

pub trait MoveBucket: Copy + Send + Sync + 'static {
    fn num_buckets(&self) -> usize;

    fn get(&self, pos: &Position, mov: Move) -> usize;
}

impl<T: SquareTransform, B: MoveBucket> ChessMoveMapper<T, B> {
    pub fn map(&self, pos: &Position, mov: Move) -> usize {
        let flip = if pos.stm() == Side::BLACK { 56 } else { 0 };
        let src = self.transform.apply(pos, usize::from(mov.src() ^ flip));
        let dst = self.transform.apply(pos, usize::from(mov.to() ^ flip));

        assert!(src < 64 && dst < 64);

        let idx = if mov.is_promo() {
            let ffile = src % 8;
            let tfile = dst % 8;
            let promo_id = 2 * ffile + tfile;

            OFFSETS[64] + 22 * (mov.promo_pc() - Piece::KNIGHT) + promo_id
        } else {
            let below = ALL_DESTINATIONS[src] & ((1 << dst) - 1);
            OFFSETS[src] + below.count_ones() as usize
        };

        self.bucket.get(pos, mov) * NUM_MOVES + idx
    }

    pub fn num_move_indices(&self) -> usize {
        NUM_MOVES * self.bucket.num_buckets()
    }
}

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

const OFFSETS: [usize; 65] = {
    let mut offsets = [0; 65];

    let mut curr = 0;
    let mut sq = 0;

    while sq < 64 {
        offsets[sq] = curr;
        curr += ALL_DESTINATIONS[sq].count_ones() as usize;
        sq += 1;
    }

    offsets[64] = curr;

    offsets
};

const ALL_DESTINATIONS: [u64; 64] = init!(|sq, 64| {
    let rank = sq / 8;
    let file = sq % 8;

    let rooks = (0xFF << (rank * 8)) ^ (A << file);
    let bishops = DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank];

    rooks | bishops | KNIGHT[sq] | KING[sq]
});

const A: u64 = 0x0101_0101_0101_0101;
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

const KNIGHT: [u64; 64] = init!(|sq, 64| {
    let n = 1 << sq;
    let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
    let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
});

const KING: [u64; 64] = init!(|sq, 64| {
    let mut k = 1 << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & !A) >> 1) | ((k & !H) << 1);
    k ^ (1 << sq)
});

const SEE_VALS: [i32; 8] = [0, 0, 100, 450, 450, 650, 1250, 0];

fn gain(pos: &Position, mov: &Move) -> i32 {
    if mov.is_en_passant() {
        return SEE_VALS[Piece::PAWN];
    }
    let mut score = SEE_VALS[pos.get_pc(1 << mov.to())];
    if mov.is_promo() {
        score += SEE_VALS[mov.promo_pc()] - SEE_VALS[Piece::PAWN];
    }
    score
}

#[allow(unused)]
fn see(pos: &Position, mov: &Move, threshold: i32) -> bool {
    let sq = usize::from(mov.to());
    assert!(sq < 64, "wha");
    let mut next = if mov.is_promo() { mov.promo_pc() } else { pos.get_pc(1 << mov.src()) };
    let mut score = gain(pos, mov) - threshold - SEE_VALS[next];

    if score >= 0 {
        return true;
    }

    let mut occ = (pos.piece(Side::WHITE) | pos.piece(Side::BLACK)) ^ (1 << mov.src()) ^ (1 << sq);
    if mov.is_en_passant() {
        occ ^= 1 << (sq ^ 8);
    }

    let bishops = pos.piece(Piece::BISHOP) | pos.piece(Piece::QUEEN);
    let rooks = pos.piece(Piece::ROOK) | pos.piece(Piece::QUEEN);
    let mut us = pos.stm() ^ 1;
    let mut attackers = (Attacks::knight(sq) & pos.piece(Piece::KNIGHT))
        | (Attacks::king(sq) & pos.piece(Piece::KING))
        | (Attacks::pawn(sq, Side::WHITE) & pos.piece(Piece::PAWN) & pos.piece(Side::BLACK))
        | (Attacks::pawn(sq, Side::BLACK) & pos.piece(Piece::PAWN) & pos.piece(Side::WHITE))
        | (Attacks::rook(sq, occ) & rooks)
        | (Attacks::bishop(sq, occ) & bishops);

    loop {
        let our_attackers = attackers & pos.piece(us);
        if our_attackers == 0 {
            break;
        }

        for pc in Piece::PAWN..=Piece::KING {
            let board = our_attackers & pos.piece(pc);
            if board > 0 {
                occ ^= board & board.wrapping_neg();
                next = pc;
                break;
            }
        }

        if [Piece::PAWN, Piece::BISHOP, Piece::QUEEN].contains(&next) {
            attackers |= Attacks::bishop(sq, occ) & bishops;
        }
        if [Piece::ROOK, Piece::QUEEN].contains(&next) {
            attackers |= Attacks::rook(sq, occ) & rooks;
        }

        attackers &= occ;
        score = -score - 1 - SEE_VALS[next];
        us ^= 1;

        if score >= 0 {
            if next == Piece::KING && attackers & pos.piece(us) > 0 {
                us ^= 1;
            }
            break;
        }
    }

    pos.stm() != us
}
