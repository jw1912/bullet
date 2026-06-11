mod direct;
mod montybinpack;
mod rng;
pub mod sfbinpack;
mod text;
pub mod viribinpack;

pub use direct::{CanBeDirectlySequentiallyLoaded, DirectSequentialDataLoader};
pub use montybinpack::MontyBinpackLoader;
pub use sfbinpack::SfBinpackLoader;
pub use text::InMemoryTextLoader;
pub use viribinpack::{ViriBinpackLoader, ViriFilter};

use bulletformat::BulletFormat;

unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::ChessBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::AtaxxBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::CudADFormat {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::MarlinFormat {}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

pub trait LoadableDataType: Sized {
    fn score(&self) -> i16;

    fn result(&self) -> GameResult;
}

impl<T: BulletFormat + 'static> LoadableDataType for T {
    fn result(&self) -> GameResult {
        [GameResult::Loss, GameResult::Draw, GameResult::Win][self.result_idx()]
    }

    fn score(&self) -> i16 {
        <Self as BulletFormat>::score(self)
    }
}

/// Dictates how data is read from a file into the expected datatype.
/// This allows for the file format to be divorced from the training
/// data format.
pub trait DataLoader<T>: Clone + Send + Sync + 'static {
    fn data_file_paths(&self) -> &[String];

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_chunks<F: FnMut(&[T]) -> bool>(&self, start_position: usize, f: F);
}
