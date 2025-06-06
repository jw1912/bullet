use bullet_lib::game::{formats::bulletformat::ChessBoard, outputs::OutputBuckets};

pub const NUM_OUTPUT_BUCKETS: usize = 8;

#[derive(Clone, Copy, Default)]
pub struct CustomOutputBuckets;
impl OutputBuckets<ChessBoard> for CustomOutputBuckets {
    const BUCKETS: usize = NUM_OUTPUT_BUCKETS;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let divisor = 32usize.div_ceil(NUM_OUTPUT_BUCKETS);
        let piece_count = pos.occ().count_ones() as u8;
        (piece_count - 2) / divisor as u8
    }
}
