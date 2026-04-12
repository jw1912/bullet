use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::tensor::TValue;
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    runtime::{Device, Gpu, Stream},
};

use crate::model::TensorMap;

#[derive(Debug)]
pub enum DataLoadingError {
    TooManyBatchesReceived,
    NoBatchesReceived,
    Message(String),
}

pub trait DataLoader: Send + Sync + 'static {
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(self, batch_size: usize, f: F) -> Result<(), DataLoadingError>;
}

pub struct PreparedBatchHost {
    pub batch_size: usize,
    pub inputs: BTreeMap<String, TValue>,
}

impl PreparedBatchHost {
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
