use std::{iter::Zip, marker::PhantomData, slice::ChunksMut};

use bullet_compiler::tensor::TValue;

use crate::Shape;

pub trait DataReader<T> {
    fn read_chunks<F: FnMut(&[T]) -> bool>(&self, skip_count: usize, f: F);
}

pub fn map_batches<T: Clone, R>(
    reader: &impl DataReader<T>,
    inputs: &TrainerInputs<T>,
    start_batch: usize,
    batch_size: usize,
    threads: u8,
    mut f: impl FnMut(Vec<TValue>) -> bool,
) {
    let mut batch = start_batch;
    let mut incomplete_buf = Vec::new();

    reader.read_chunks(start_batch * batch_size, |chunk| {
        let remainder = if !incomplete_buf.is_empty() {
            let remainder = batch_size - incomplete_buf.len();

            if chunk.len() >= remainder {
                incomplete_buf.extend_from_slice(&chunk[..remainder]);
                let prepared = inputs.prepare(&incomplete_buf, batch, threads);
                batch += 1;

                if f(prepared) {
                    return true;
                }

                incomplete_buf.clear();
            } else {
                incomplete_buf.extend_from_slice(chunk);
            }

            remainder
        } else {
            0
        };

        if chunk.len() >= remainder {
            let chunks = chunk[remainder..chunk.len()].chunks_exact(batch_size);
            incomplete_buf.extend_from_slice(chunks.remainder());

            for data in chunks {
                let prepared = inputs.prepare(data, batch, threads);
                batch += 1;

                if f(prepared) {
                    return true;
                }
            }
        }

        false
    });
}

pub struct TrainerInputsBuilder<T> {
    inputs: T,
}

impl Default for TrainerInputsBuilder<()> {
    fn default() -> Self {
        Self { inputs: () }
    }
}

impl TrainerInputsBuilder<()> {
    pub fn add_input<T: InputType>(self, input: T) -> TrainerInputsBuilder<T> {
        TrainerInputsBuilder { inputs: input }
    }
}

impl<T: InputType + 'static> TrainerInputsBuilder<T> {
    pub fn add_input<U: InputType>(self, input: U) -> TrainerInputsBuilder<(T, U)> {
        TrainerInputsBuilder { inputs: (self.inputs, input) }
    }

    pub fn build<P: Send + Sync, F>(self, f: F) -> TrainerInputs<P>
    where
        F: for<'a> Fn(&P, usize, T::Slices<'a>) + Send + Sync + 'static,
    {
        let inputs = self.inputs;

        let func = move |batch: &[P], batch_number, threads| {
            let f = &f;
            let chunk_size = batch.len().div_ceil(usize::from(threads));

            let mut bufs = inputs.make_bufs(batch.len());
            let chunks = inputs.chunks(&mut bufs, chunk_size);

            std::thread::scope(|s| {
                let inputs = &inputs;

                for (data, chunk) in batch.chunks(chunk_size).zip(chunks) {
                    s.spawn(move || {
                        for (datapoint, slices) in data.iter().zip(inputs.slices(chunk)) {
                            f(datapoint, batch_number, slices);
                        }
                    });
                }
            });

            let mut vec = Vec::new();
            inputs.append_bufs_to_vec(bufs, &mut vec);
            vec
        };

        TrainerInputs { func: Box::new(func) }
    }
}

pub struct TrainerInputs<T> {
    #[allow(clippy::type_complexity)]
    func: Box<dyn Fn(&[T], usize, u8) -> Vec<TValue>>,
}

impl<T> TrainerInputs<T> {
    pub fn prepare(&self, data: &[T], batch_number: usize, threads: u8) -> Vec<TValue> {
        (self.func)(data, batch_number, threads)
    }
}

pub trait InputType: Send + Sync {
    type Buf: Send + Sync;
    type Chunks<'a>: 'a + Iterator<Item = Self::Slices<'a>> + Send + Sync;
    type Slices<'a>: 'a + Send + Sync;

    fn make_bufs(&self, batch_size: usize) -> Self::Buf;

    fn chunks<'a>(&self, buf: &'a mut Self::Buf, chunk_size: usize) -> Self::Chunks<'a>;

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a>;

    fn append_bufs_to_vec(&self, buf: Self::Buf, vec: &mut Vec<TValue>);
}

#[derive(Clone, Copy)]
pub struct SparseInput {
    nnz: usize,
    shape: Shape,
}

impl SparseInput {
    pub fn new(shape: impl Into<Shape>, nnz: usize) -> Self {
        Self { nnz, shape: shape.into() }
    }
}

#[derive(Clone, Copy)]
pub struct DenseInput<T> {
    shape: Shape,
    phantom: PhantomData<T>,
}
impl<T> DenseInput<T> {
    pub fn new(shape: impl Into<Shape>) -> Self {
        Self { shape: shape.into(), phantom: PhantomData }
    }
}

impl InputType for SparseInput {
    type Buf = Vec<i32>;
    type Chunks<'a> = ChunksMut<'a, i32>;
    type Slices<'a> = &'a mut [i32];

    fn make_bufs(&self, batch_size: usize) -> Self::Buf {
        vec![0; self.nnz * batch_size]
    }

    fn chunks<'a>(&self, buf: &'a mut Self::Buf, chunk_size: usize) -> Self::Chunks<'a> {
        buf.chunks_mut(chunk_size * self.nnz)
    }

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a> {
        chunk.chunks_mut(self.shape.size())
    }

    fn append_bufs_to_vec(&self, buf: Self::Buf, vec: &mut Vec<TValue>) {
        vec.push(TValue::I32(buf));
    }
}

impl InputType for DenseInput<f32> {
    type Buf = Vec<f32>;
    type Chunks<'a> = ChunksMut<'a, f32>;
    type Slices<'a> = &'a mut [f32];

    fn make_bufs(&self, batch_size: usize) -> Self::Buf {
        vec![0.0; self.shape.size() * batch_size]
    }

    fn chunks<'a>(&self, buf: &'a mut Self::Buf, chunk_size: usize) -> Self::Chunks<'a> {
        buf.chunks_mut(chunk_size * self.shape.size())
    }

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a> {
        chunk.chunks_mut(self.shape.size())
    }

    fn append_bufs_to_vec(&self, buf: Self::Buf, vec: &mut Vec<TValue>) {
        vec.push(TValue::F32(buf));
    }
}

impl<T: InputType, U: InputType> InputType for (T, U) {
    type Buf = (T::Buf, U::Buf);
    type Chunks<'a> = Zip<T::Chunks<'a>, U::Chunks<'a>>;
    type Slices<'a> = (T::Slices<'a>, U::Slices<'a>);

    fn make_bufs(&self, batch_size: usize) -> Self::Buf {
        (self.0.make_bufs(batch_size), self.1.make_bufs(batch_size))
    }

    fn chunks<'a>(&self, buf: &'a mut Self::Buf, chunk_size: usize) -> Self::Chunks<'a> {
        self.0.chunks(&mut buf.0, chunk_size).zip(self.1.chunks(&mut buf.1, chunk_size))
    }

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a> {
        self.0.slices(chunk.0).zip(self.1.slices(chunk.1))
    }

    fn append_bufs_to_vec(&self, buf: Self::Buf, vec: &mut Vec<TValue>) {
        self.0.append_bufs_to_vec(buf.0, vec);
        self.1.append_bufs_to_vec(buf.1, vec);
    }
}
