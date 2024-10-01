mod direct_sequential;

use bulletformat::BulletFormat;

pub use direct_sequential::DirectSequentialDataLoader;

use crate::{inputs::InputType, outputs::OutputBuckets, trainer::{DataPreparer, DefaultDataPreparer}};

pub trait DataLoader<T>: Clone + Send + Sync + 'static {
    fn data_file_paths(&self) -> &[String];

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[T]) -> bool>(&self, batch_size: usize, f: F);
}

#[derive(Clone)]
pub struct DefaultDataLoader<I, O, D> {
    input_getter: I,
    output_getter: O,
    scale: f32,
    loader: D,
}

impl<I, O, D> DataPreparer for DefaultDataLoader<I, O, D>
where I: InputType, O: OutputBuckets<I::RequiredDataType>, D: DataLoader<I::RequiredDataType>
{
    type DataType = I::RequiredDataType;
    type PreparedData = DefaultDataPreparer<I, O>;

    fn get_data_file_paths(&self) -> &[String] {
        self.loader.data_file_paths()
    }

    fn try_count_positions(&self) -> Option<u64> {
        self.loader.count_positions()
    }

    fn load_and_map_batches<F: FnMut(&[Self::DataType]) -> bool>(&self, batch_size: usize, f: F) {
        self.loader.map_batches(batch_size, f);
    }

    fn prepare(&self, data: &[Self::DataType], threads: usize, blend: f32) -> Self::PreparedData {
        DefaultDataPreparer::prepare(
            self.input_getter,
            self.output_getter,
            data,
            threads,
            blend,
            self.scale,
        )
    }
}

/// ### Safety
/// `&[T]` must be reinterpretable as `&[U]`
unsafe fn to_slice_with_lifetime<T, U>(slice: &[T]) -> &[U] {
    let src_size = std::mem::size_of_val(slice);
    let tgt_size = std::mem::size_of::<U>();

    assert!(src_size % tgt_size == 0, "Target type size does not divide slice size!");

    let len = src_size / tgt_size;
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), len) }
}
