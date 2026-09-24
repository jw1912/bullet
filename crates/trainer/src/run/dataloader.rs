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
    free: Mutex<BTreeMap<usize, Vec<Vec<T>>>>,
}

impl<T> Pool<T> {
    fn give(&self, value: Vec<T>) {
        self.free.lock().unwrap().entry(value.len().next_power_of_two()).or_default().push(value);
    }
}

impl<E: Clone + Default> Pool<E> {
    fn take_vec(&self, len: usize) -> Vec<E> {
        let capacity = len.next_power_of_two();
        let cached = self.free.lock().unwrap().get_mut(&capacity).and_then(Vec::pop);
        let mut value = cached.unwrap_or_default();

        if len > value.capacity() {
            value.reserve_exact(len - value.len());
        }
        value.resize(len, E::default());
        value
    }
}

#[derive(Default)]
pub struct HostPool {
    i32s: Pool<i32>,
    f32s: Pool<f32>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_pool_reuses_matching_sizes_regardless_of_return_order() {
        let pool = HostPool::default();
        let mut small = pool.take_i32(1);
        let mut large = pool.take_i32(4096);
        small[0] = 7;
        large[4095] = 11;
        let pointers = [small.as_ptr(), large.as_ptr()];
        pool.give(TValue::I32(small));
        pool.give(TValue::I32(large));
        let small = pool.take_i32(1);
        let large = pool.take_i32(4096);
        assert_eq!([small.as_ptr(), large.as_ptr()], pointers);
        assert_eq!(small, [7]);
        assert_eq!(large[4095], 11);
        assert_eq!(small.capacity(), 1);
        // A new shape is correctly sized and initialised without consuming a
        // differently sized cached buffer. Dtypes have separate pools.
        pool.give(TValue::I32(large));
        assert_eq!(pool.take_i32(2), [0, 0]);
        assert_eq!(pool.take_f32(4096), vec![0.0; 4096]);
        let reused = pool.take_i32(4096);
        assert_eq!(reused.as_ptr(), pointers[1]);
        let mut near_size = pool.take_i32(5);
        near_size.fill(9);
        pool.give(TValue::I32(near_size));
        let grown = pool.take_i32(7);
        assert_eq!(grown, [9, 9, 9, 9, 9, 0, 0]);
    }
}
