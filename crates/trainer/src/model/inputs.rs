use std::{iter::Zip, marker::PhantomData, slice::ChunksMut, sync::Arc};

use bullet_compiler::{
    model::{ModelBuilder, ModelNode, Shape},
    tensor::TValue,
};

use crate::run::{HostPool, PreparedBatchHost, Step};

pub struct ModelInputs<T> {
    inputs: T,
    names: Vec<String>,
}

impl Default for ModelInputs<()> {
    fn default() -> Self {
        Self { inputs: (), names: Vec::new() }
    }
}

impl ModelInputs<()> {
    fn add<T: InputType>(self, name: String, input: T) -> ModelInputs<T> {
        ModelInputs { inputs: input, names: vec![name] }
    }

    pub fn add_dense(self, name: impl Into<String>, shape: impl Into<Shape>) -> ModelInputs<DenseInput<f32>> {
        let name = name.into();
        self.add(name.clone(), DenseInput::<f32>::new(name, shape))
    }

    pub fn add_sparse(self, name: impl Into<String>, shape: impl Into<Shape>, nnz: usize) -> ModelInputs<SparseInput> {
        let name = name.into();
        self.add(name.clone(), SparseInput::new(name, shape, nnz))
    }
}

impl<T: InputType + 'static> ModelInputs<T> {
    fn add<U: InputType>(self, name: impl Into<String>, input: U) -> ModelInputs<(T, U)> {
        let mut names = self.names;
        names.push(name.into());
        ModelInputs { inputs: (self.inputs, input), names }
    }

    pub fn add_dense(self, name: impl Into<String>, shape: impl Into<Shape>) -> ModelInputs<(T, DenseInput<f32>)> {
        let name = name.into();
        self.add(name.clone(), DenseInput::<f32>::new(name.clone(), shape))
    }

    pub fn add_sparse(
        self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        nnz: usize,
    ) -> ModelInputs<(T, SparseInput)> {
        let name = name.into();
        self.add(name.clone(), SparseInput::new(name, shape, nnz))
    }

    pub fn make_nodes<'a>(&self, builder: &'a ModelBuilder) -> T::Nodes<'a> {
        self.inputs.make_nodes(builder)
    }
}

pub struct ModelInputsMapper<T> {
    #[allow(clippy::type_complexity)]
    func: Arc<dyn Fn(&[T], Step, u8) -> PreparedBatchHost + Send + Sync>,
    names: Vec<String>,
}

impl<T> Clone for ModelInputsMapper<T> {
    fn clone(&self) -> Self {
        Self { func: self.func.clone(), names: self.names.clone() }
    }
}

impl<T: Send + Sync> ModelInputsMapper<T> {
    pub fn build<I: InputType, F>(inputs: &ModelInputs<I>, f: F) -> Self
    where
        F: for<'a> Fn(&T, Step, I::Slices<'a>) + Send + Sync + 'static,
    {
        let names = inputs.names.clone();
        let inp = inputs.inputs.clone();

        let pool = Arc::<HostPool>::default();

        let func = move |batch: &[T], step, threads| {
            let f = &f;
            let chunk_size = batch.len().div_ceil(usize::from(threads));

            let mut bufs = inp.make_bufs(batch.len(), &pool);
            let chunks = inp.chunks(&mut bufs, chunk_size);

            std::thread::scope(|s| {
                let inputs = &inp;

                for (data, chunk) in batch.chunks(chunk_size).zip(chunks) {
                    s.spawn(move || {
                        for (datapoint, slices) in data.iter().zip(inputs.slices(chunk)) {
                            f(datapoint, step, slices);
                        }
                    });
                }
            });

            let mut vec = Vec::new();
            inp.append_bufs_to_vec(bufs, &mut vec);
            let inputs = names.iter().cloned().zip(vec).collect();
            PreparedBatchHost::new(pool.clone(), inputs)
        };

        ModelInputsMapper { func: Arc::new(func), names: inputs.names.clone() }
    }

    pub fn map(&self, data: &[T], step: Step, threads: u8) -> PreparedBatchHost {
        (self.func)(data, step, threads)
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }
}

pub trait InputType: Clone + Send + Sync + 'static {
    type Buf: Send + Sync;
    type Chunks<'a>: 'a + Iterator<Item = Self::Slices<'a>> + Send + Sync;
    type Slices<'a>: 'a + Send + Sync;
    type Nodes<'a>: 'a;

    fn make_bufs(&self, batch_size: usize, pool: &Arc<HostPool>) -> Self::Buf;

    fn chunks<'a>(&self, buf: &'a mut Self::Buf, chunk_size: usize) -> Self::Chunks<'a>;

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a>;

    fn make_nodes<'a>(&self, builder: &'a ModelBuilder) -> Self::Nodes<'a>;

    fn append_bufs_to_vec(&self, buf: Self::Buf, vec: &mut Vec<TValue>);
}

#[derive(Clone)]
pub struct SparseInput {
    name: String,
    shape: Shape,
    nnz: usize,
}

impl SparseInput {
    pub fn new(name: String, shape: impl Into<Shape>, nnz: usize) -> Self {
        Self { name, shape: shape.into(), nnz }
    }
}

#[derive(Clone)]
pub struct DenseInput<T> {
    name: String,
    shape: Shape,
    phantom: PhantomData<T>,
}
impl<T> DenseInput<T> {
    pub fn new(name: String, shape: impl Into<Shape>) -> Self {
        Self { name, shape: shape.into(), phantom: PhantomData }
    }
}

impl InputType for SparseInput {
    type Buf = Vec<i32>;
    type Chunks<'a> = ChunksMut<'a, i32>;
    type Slices<'a> = &'a mut [i32];
    type Nodes<'a> = ModelNode<'a>;

    fn make_bufs(&self, batch_size: usize, pool: &Arc<HostPool>) -> Self::Buf {
        pool.take_i32(self.nnz * batch_size)
    }

    fn chunks<'a>(&self, buf: &'a mut Self::Buf, chunk_size: usize) -> Self::Chunks<'a> {
        buf.chunks_mut(chunk_size * self.nnz)
    }

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a> {
        chunk.chunks_mut(self.nnz)
    }

    fn make_nodes<'a>(&self, builder: &'a ModelBuilder) -> Self::Nodes<'a> {
        builder.new_sparse_input(self.name.clone(), self.shape, self.nnz)
    }

    fn append_bufs_to_vec(&self, buf: Self::Buf, vec: &mut Vec<TValue>) {
        vec.push(TValue::I32(buf));
    }
}

impl InputType for DenseInput<f32> {
    type Buf = Vec<f32>;
    type Chunks<'a> = ChunksMut<'a, f32>;
    type Slices<'a> = &'a mut [f32];
    type Nodes<'a> = ModelNode<'a>;

    fn make_bufs(&self, batch_size: usize, pool: &Arc<HostPool>) -> Self::Buf {
        let mut buf = pool.take_f32(self.shape.size() * batch_size);
        buf.fill(0.0);
        buf
    }

    fn chunks<'a>(&self, buf: &'a mut Self::Buf, chunk_size: usize) -> Self::Chunks<'a> {
        buf.chunks_mut(chunk_size * self.shape.size())
    }

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a> {
        chunk.chunks_mut(self.shape.size())
    }

    fn make_nodes<'a>(&self, builder: &'a ModelBuilder) -> Self::Nodes<'a> {
        builder.new_dense_input(self.name.clone(), self.shape)
    }

    fn append_bufs_to_vec(&self, buf: Self::Buf, vec: &mut Vec<TValue>) {
        vec.push(TValue::F32(buf));
    }
}

impl<T: InputType, U: InputType> InputType for (T, U) {
    type Buf = (T::Buf, U::Buf);
    type Chunks<'a> = Zip<T::Chunks<'a>, U::Chunks<'a>>;
    type Slices<'a> = (T::Slices<'a>, U::Slices<'a>);
    type Nodes<'a> = (T::Nodes<'a>, U::Nodes<'a>);

    fn make_bufs(&self, batch_size: usize, pool: &Arc<HostPool>) -> Self::Buf {
        (self.0.make_bufs(batch_size, pool), self.1.make_bufs(batch_size, pool))
    }

    fn chunks<'a>(&self, buf: &'a mut Self::Buf, chunk_size: usize) -> Self::Chunks<'a> {
        self.0.chunks(&mut buf.0, chunk_size).zip(self.1.chunks(&mut buf.1, chunk_size))
    }

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a> {
        self.0.slices(chunk.0).zip(self.1.slices(chunk.1))
    }

    fn make_nodes<'a>(&self, builder: &'a ModelBuilder) -> Self::Nodes<'a> {
        (self.0.make_nodes(builder), self.1.make_nodes(builder))
    }

    fn append_bufs_to_vec(&self, buf: Self::Buf, vec: &mut Vec<TValue>) {
        self.0.append_bufs_to_vec(buf.0, vec);
        self.1.append_bufs_to_vec(buf.1, vec);
    }
}
