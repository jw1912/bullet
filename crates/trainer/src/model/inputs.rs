use std::{iter::Zip, marker::PhantomData, slice::ChunksMut};

use bullet_compiler::{model::Shape, tensor::TValue};

pub struct ModelInputsBuilder<T> {
    inputs: T,
    names: Vec<String>,
}

impl Default for ModelInputsBuilder<()> {
    fn default() -> Self {
        Self { inputs: (), names: Vec::new() }
    }
}

impl ModelInputsBuilder<()> {
    fn add_input<T: InputType>(self, name: impl Into<String>, input: T) -> ModelInputsBuilder<T> {
        ModelInputsBuilder { inputs: input, names: vec![name.into()] }
    }

    pub fn add_dense_input(
        self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
    ) -> ModelInputsBuilder<DenseInput<f32>> {
        self.add_input(name, DenseInput::<f32>::new(shape))
    }

    pub fn add_sparse_input(
        self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        nnz: usize,
    ) -> ModelInputsBuilder<SparseInput> {
        self.add_input(name, SparseInput::new(shape, nnz))
    }
}

impl<T: InputType + 'static> ModelInputsBuilder<T> {
    fn add_input<U: InputType>(self, name: impl Into<String>, input: U) -> ModelInputsBuilder<(T, U)> {
        let mut names = self.names;
        names.push(name.into());
        ModelInputsBuilder { inputs: (self.inputs, input), names }
    }

    pub fn add_dense_input(
        self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
    ) -> ModelInputsBuilder<(T, DenseInput<f32>)> {
        self.add_input(name, DenseInput::<f32>::new(shape))
    }

    pub fn add_sparse_input(
        self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        nnz: usize,
    ) -> ModelInputsBuilder<(T, SparseInput)> {
        self.add_input(name, SparseInput::new(shape, nnz))
    }

    pub fn build<P: Send + Sync, F>(self, f: F) -> ModelInputs<P, T>
    where
        F: for<'a> Fn(&P, usize, T::Slices<'a>) + Send + Sync + 'static,
    {
        let inputs = self.inputs.clone();

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

        ModelInputs { inputs: self.inputs, mapper: ModelInputsMapper { func: Box::new(func), names: self.names } }
    }
}

pub struct ModelInputs<D, T> {
    inputs: T,
    mapper: ModelInputsMapper<D>,
}

impl<D, T> ModelInputs<D, T> {
    pub fn inputs(&self) -> &T {
        &self.inputs
    }

    pub fn into_mapper(self) -> ModelInputsMapper<D> {
        self.mapper
    }
}

pub struct ModelInputsMapper<T> {
    #[allow(clippy::type_complexity)]
    func: Box<dyn Fn(&[T], usize, u8) -> Vec<TValue>>,
    names: Vec<String>,
}

impl<T> ModelInputsMapper<T> {
    pub fn map(&self, data: &[T], batch_number: usize, threads: u8) -> Vec<TValue> {
        (self.func)(data, batch_number, threads)
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }
}

pub trait InputType: Clone + Send + Sync {
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
