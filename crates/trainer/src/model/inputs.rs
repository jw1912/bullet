use std::{iter::Zip, marker::PhantomData, slice::ChunksMut, sync::Arc};

use bullet_compiler::{
    model::{ModelBuilder, ModelNode, Shape},
    tensor::DType,
};
use bullet_gpu::runtime::Gpu;

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

/// Mutable view of a single input's host buffer for a whole batch
pub enum InputSlice<'a> {
    F32(&'a mut [f32]),
    I32(&'a mut [i32]),
}

/// Iterator over the host buffers of a batch, in input order
pub type InputSlices<'a> = std::vec::IntoIter<InputSlice<'a>>;

pub struct ModelInputsMapper<T> {
    #[allow(clippy::type_complexity)]
    func: Arc<dyn Fn(&[T], Step, u8, InputSlices<'_>) + Send + Sync>,
    names: Vec<String>,
    specs: Vec<(DType, usize)>,
}

impl<T> Clone for ModelInputsMapper<T> {
    fn clone(&self) -> Self {
        Self { func: self.func.clone(), names: self.names.clone(), specs: self.specs.clone() }
    }
}

impl<T: Send + Sync> ModelInputsMapper<T> {
    pub fn build<I: InputType, F>(inputs: &ModelInputs<I>, f: F) -> Self
    where
        F: for<'a> Fn(&T, Step, I::Slices<'a>) + Send + Sync + 'static,
    {
        let inp = inputs.inputs.clone();

        let mut specs = Vec::new();
        inp.append_specs(&mut specs);

        let func = move |batch: &[T], step, threads: u8, mut bufs: InputSlices<'_>| {
            let f = &f;
            let chunk_size = batch.len().div_ceil(usize::from(threads.max(1)));

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
        };

        ModelInputsMapper { func: Arc::new(func), names: inputs.names.clone(), specs }
    }

    /// Map the given data into pinned host buffers taken from `pool`
    pub fn map<G: Gpu>(
        &self,
        pool: &Arc<HostPool<G>>,
        data: &[T],
        step: Step,
        threads: u8,
    ) -> Result<PreparedBatchHost<G>, G::Error> {
        let mut bufs = self
            .specs
            .iter()
            .map(|&(dtype, size)| pool.take(dtype, size * data.len()))
            .collect::<Result<Vec<_>, _>>()?;

        let slices = bufs
            .iter_mut()
            .map(|buf| match buf.dtype() {
                DType::F32 => InputSlice::F32(buf.as_mut_slice().unwrap()),
                DType::I32 => InputSlice::I32(buf.as_mut_slice().unwrap()),
            })
            .collect::<Vec<_>>();

        (self.func)(data, step, threads, slices.into_iter());

        let inputs = self.names.iter().cloned().zip(bufs).collect();
        Ok(PreparedBatchHost::new(pool.clone(), inputs))
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }
}

pub trait InputType: Clone + Send + Sync + 'static {
    type Chunks<'a>: 'a + Iterator<Item = Self::Slices<'a>> + Send + Sync;
    type Slices<'a>: 'a + Send + Sync;
    type Nodes<'a>: 'a;

    /// Append the dtype and per-datapoint size of each host buffer this input requires
    fn append_specs(&self, specs: &mut Vec<(DType, usize)>);

    /// Consume the host buffers for this input (in the order given by `append_specs`)
    /// and split them into chunks of `chunk_size` datapoints
    fn chunks<'a>(&self, bufs: &mut InputSlices<'a>, chunk_size: usize) -> Self::Chunks<'a>;

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a>;

    fn make_nodes<'a>(&self, builder: &'a ModelBuilder) -> Self::Nodes<'a>;
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
    type Chunks<'a> = ChunksMut<'a, i32>;
    type Slices<'a> = &'a mut [i32];
    type Nodes<'a> = ModelNode<'a>;

    fn append_specs(&self, specs: &mut Vec<(DType, usize)>) {
        specs.push((DType::I32, self.nnz));
    }

    fn chunks<'a>(&self, bufs: &mut InputSlices<'a>, chunk_size: usize) -> Self::Chunks<'a> {
        let Some(InputSlice::I32(buf)) = bufs.next() else { panic!("Expected an I32 buffer!") };
        buf.chunks_mut(chunk_size * self.nnz)
    }

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a> {
        chunk.chunks_mut(self.nnz)
    }

    fn make_nodes<'a>(&self, builder: &'a ModelBuilder) -> Self::Nodes<'a> {
        builder.new_sparse_input(self.name.clone(), self.shape, self.nnz)
    }
}

impl InputType for DenseInput<f32> {
    type Chunks<'a> = ChunksMut<'a, f32>;
    type Slices<'a> = &'a mut [f32];
    type Nodes<'a> = ModelNode<'a>;

    fn append_specs(&self, specs: &mut Vec<(DType, usize)>) {
        specs.push((DType::F32, self.shape.size()));
    }

    fn chunks<'a>(&self, bufs: &mut InputSlices<'a>, chunk_size: usize) -> Self::Chunks<'a> {
        let Some(InputSlice::F32(buf)) = bufs.next() else { panic!("Expected an F32 buffer!") };
        buf.fill(0.0);
        buf.chunks_mut(chunk_size * self.shape.size())
    }

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a> {
        chunk.chunks_mut(self.shape.size())
    }

    fn make_nodes<'a>(&self, builder: &'a ModelBuilder) -> Self::Nodes<'a> {
        builder.new_dense_input(self.name.clone(), self.shape)
    }
}

impl<T: InputType, U: InputType> InputType for (T, U) {
    type Chunks<'a> = Zip<T::Chunks<'a>, U::Chunks<'a>>;
    type Slices<'a> = (T::Slices<'a>, U::Slices<'a>);
    type Nodes<'a> = (T::Nodes<'a>, U::Nodes<'a>);

    fn append_specs(&self, specs: &mut Vec<(DType, usize)>) {
        self.0.append_specs(specs);
        self.1.append_specs(specs);
    }

    fn chunks<'a>(&self, bufs: &mut InputSlices<'a>, chunk_size: usize) -> Self::Chunks<'a> {
        let first = self.0.chunks(bufs, chunk_size);
        first.zip(self.1.chunks(bufs, chunk_size))
    }

    fn slices<'a>(&self, chunk: <Self::Chunks<'a> as Iterator>::Item) -> Self::Chunks<'a> {
        self.0.slices(chunk.0).zip(self.1.slices(chunk.1))
    }

    fn make_nodes<'a>(&self, builder: &'a ModelBuilder) -> Self::Nodes<'a> {
        (self.0.make_nodes(builder), self.1.make_nodes(builder))
    }
}
