use bulletformat::ChessBoard;

use crate::rng;

pub trait OutputBuckets<T>: Send + Sync + Copy + Default + 'static {
    const BUCKETS: usize;

    fn bucket(&mut self, pos: &T) -> u8;
}

#[derive(Clone, Copy, Default)]
pub struct Single;
impl<T: 'static> OutputBuckets<T> for Single {
    const BUCKETS: usize = 1;

    fn bucket(&mut self, _: &T) -> u8 {
        0
    }
}

#[derive(Clone, Copy, Default)]
pub struct MaterialCount<const N: usize>;
impl<const N: usize> OutputBuckets<ChessBoard> for MaterialCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&mut self, pos: &ChessBoard) -> u8 {
        let divisor = 32usize.div_ceil(N);
        (pos.occ().count_ones() as u8 - 2) / divisor as u8
    }
}

#[derive(Clone, Copy, Default)]
pub struct MaterialCountNoisy<const N: usize> {
    rng: rng::SimpleRand,
}
impl<const N: usize> OutputBuckets<ChessBoard> for MaterialCountNoisy<N> {
    const BUCKETS: usize = N;

    fn bucket(&mut self, pos: &ChessBoard) -> u8 {
        // 3% of the time, hop one bucket
        // 0.1% of the time, hop two buckets
        const THRESHOLD_ONE: u64 = 17874895007424555515; // 96.9% of u64::MAX
        const THRESHOLD_TWO: u64 = 18428297329635842064; // 99.9$ of u64::MAX

        let divisor = 32usize.div_ceil(N);
        let bucket = (pos.occ().count_ones() as u8 - 2) / divisor as u8;

        let r = self.rng.rng();
        // determine offset size
        let offset = match () {
            () if r < THRESHOLD_ONE => 0,
            () if r < THRESHOLD_TWO => 1,
            () => 2,
        };
        // go up half the time, go down half the time
        let multiplier = 1 - ((r & 1) as i32 * 2);

        let new_bucket = i32::from(bucket) + offset * multiplier;

        // get extremal values back in the valid range
        new_bucket.clamp(0, N as i32) as u8
    }
}
