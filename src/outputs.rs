#![allow(unused)]

use bulletformat::ChessBoard;

pub struct Single;
impl Single {
    pub const NUM: usize = 1;

    pub fn get_bucket(pos: &ChessBoard) -> usize {
        0
    }
}

pub struct MaterialCount<const N: usize>;
impl<const N: usize> MaterialCount<N> {
    pub const NUM: usize = N;

    const DIVISOR: usize = 32 / Self::NUM;

    pub fn get_bucket(pos: &ChessBoard) -> usize {
        (pos.occ().count_ones() as usize - 2) / Self::DIVISOR
    }
}
