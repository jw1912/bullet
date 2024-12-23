pub use crate::trainer::default::loader::{
    CanBeDirectlySequentiallyLoaded, DirectSequentialDataLoader, SfBinpackLoader,
};

/// Dictates how data is read from a file into the expected datatype.
/// This allows for the file format to be divorced from the training
/// data format.
pub trait DataLoader<T>: Clone + Send + Sync + 'static {
    fn data_file_paths(&self) -> &[String];

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[T]) -> bool>(&self, batch_size: usize, f: F);
}
