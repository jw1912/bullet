use std::{
    collections::{BTreeMap, HashMap},
    mem,
    sync::{Arc, Mutex},
};

use bullet_compiler::tensor::DType;
use bullet_gpu::{
    buffer::{Buffer, PinnedBuffer, SyncOnValue},
    runtime::{Device, Gpu, Stream},
};

use crate::{model::TensorMap, run::Step};

#[derive(Debug)]
pub enum DataLoadingError {
    TooManyBatchesReceived,
    NoBatchesReceived,
    Message(String),
}

pub trait DataLoader: Send + Sync + 'static {
    fn map_batches<G: Gpu, F: FnMut(PreparedBatchHost<G>) -> bool>(
        self,
        pool: &Arc<HostPool<G>>,
        start: Step,
        batch_size: usize,
        f: F,
    ) -> Result<(), DataLoadingError>;
}

pub struct PreparedBatchHost<G: Gpu> {
    pub inputs: BTreeMap<String, PinnedBuffer<G>>,
    pool: Arc<HostPool<G>>,
}

impl<G: Gpu> PreparedBatchHost<G> {
    pub fn new(pool: Arc<HostPool<G>>, inputs: BTreeMap<String, PinnedBuffer<G>>) -> Self {
        Self { inputs, pool }
    }

    pub fn copy_to_device_async<'a>(
        &'a self,
        stream: &Arc<Stream<G>>,
        tensors: &TensorMap<G>,
    ) -> Result<Vec<SyncOnValue<G, &'a PinnedBuffer<G>>>, G::Error> {
        let mut syncs = Vec::new();

        for (id, tensor) in tensors {
            let value = self.inputs.get(id).ok_or("Missing input!".into())?;
            syncs.push(tensor.copy_from_pinned_async(stream, value)?);
        }

        Ok(syncs)
    }

    pub fn to_device(self, device: &Arc<Device<G>>) -> Result<TensorMap<G>, G::Error> {
        self.inputs
            .iter()
            .map(|(id, value)| Buffer::from_pinned(device, value).map(|tensor| (id.clone(), tensor)))
            .collect()
    }
}

impl<G: Gpu> Drop for PreparedBatchHost<G> {
    fn drop(&mut self) {
        for (_, value) in mem::take(&mut self.inputs) {
            self.pool.give(value);
        }
    }
}

type FreeList<G> = HashMap<(DType, usize), Vec<PinnedBuffer<G>>>;

/// Pool of pinned host buffers for preparing batches in, as pinned
/// allocations are expensive and batches are generally the same size
pub struct HostPool<G: Gpu> {
    device: Arc<Device<G>>,
    free: Mutex<FreeList<G>>,
}

impl<G: Gpu> HostPool<G> {
    pub fn new(device: Arc<Device<G>>) -> Arc<Self> {
        Arc::new(Self { device, free: Mutex::default() })
    }

    pub fn device(&self) -> Arc<Device<G>> {
        self.device.clone()
    }

    /// Take a buffer of the given dtype and size from the pool, allocating
    /// a new one if needed. Contents of reused buffers are unspecified.
    pub fn take(&self, dtype: DType, size: usize) -> Result<PinnedBuffer<G>, G::Error> {
        let capacity = size.next_power_of_two();
        let cached = self.free.lock().unwrap().get_mut(&(dtype, capacity)).and_then(Vec::pop);

        let mut value = match cached {
            Some(value) => value,
            None => PinnedBuffer::zeroed(&self.device, dtype, capacity)?,
        };

        value.set_size(size)?;
        Ok(value)
    }

    fn give(&self, value: PinnedBuffer<G>) {
        self.free.lock().unwrap().entry((value.dtype(), value.capacity())).or_default().push(value);
    }
}
