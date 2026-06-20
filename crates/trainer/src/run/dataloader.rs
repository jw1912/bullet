use std::{
    collections::BTreeMap,
    mem,
    sync::{Arc, Mutex},
};

use bullet_compiler::tensor::TValue;
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
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
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        start: Step,
        batch_size: usize,
        f: F,
    ) -> Result<(), DataLoadingError>;
}

pub struct PreparedBatchHost {
    pub inputs: BTreeMap<String, TValue>,
    pool: Arc<HostPool>,
}

impl PreparedBatchHost {
    pub fn new(pool: Arc<HostPool>, inputs: BTreeMap<String, TValue>) -> Self {
        Self { inputs, pool }
    }

    pub fn copy_to_device_async<'a, G: Gpu>(
        &'a self,
        stream: &Arc<Stream<G>>,
        tensors: &TensorMap<G>,
    ) -> Result<Vec<SyncOnValue<G, &'a TValue>>, G::Error> {
        let mut syncs = Vec::new();

        for (id, tensor) in tensors {
            let value = self.inputs.get(id).ok_or("Missing input!".into())?;
            syncs.push(tensor.copy_from_host_async(stream, value)?);
        }

        Ok(syncs)
    }

    pub fn to_device<G: Gpu>(self, device: &Arc<Device<G>>) -> Result<TensorMap<G>, G::Error> {
        self.inputs
            .iter()
            .map(|(id, value)| Buffer::from_host(device, value).map(|tensor| (id.clone(), tensor)))
            .collect()
    }
}

impl Drop for PreparedBatchHost {
    fn drop(&mut self) {
        for (_, value) in mem::take(&mut self.inputs) {
            self.pool.give(value);
        }
    }
}

#[derive(Default)]
struct Pool<T> {
    free: Mutex<Vec<T>>,
}

impl<T> Pool<T> {
    fn take(&self) -> Option<T> {
        self.free.lock().unwrap().pop()
    }

    fn give(&self, value: T) {
        self.free.lock().unwrap().push(value);
    }
}

impl<E: Clone + Default> Pool<Vec<E>> {
    fn take_vec(&self, len: usize) -> Vec<E> {
        let mut value = self.take().unwrap_or_default();
        value.resize(len, E::default());
        value
    }
}

#[derive(Default)]
pub struct HostPool {
    i32s: Pool<Vec<i32>>,
    f32s: Pool<Vec<f32>>,
}

impl HostPool {
    pub fn take_i32(&self, len: usize) -> Vec<i32> {
        self.i32s.take_vec(len)
    }

    pub fn take_f32(&self, len: usize) -> Vec<f32> {
        self.f32s.take_vec(len)
    }

    fn give(&self, value: TValue) {
        match value {
            TValue::I32(v) => self.i32s.give(v),
            TValue::F32(v) => self.f32s.give(v),
        }
    }
}
