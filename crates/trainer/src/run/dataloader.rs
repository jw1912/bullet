use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::tensor::TValue;
use bullet_gpu::{
    buffer::Buffer,
    runtime::{Gpu, Stream},
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
    pub fn to_device_blocking<G: Gpu>(self, stream: &Arc<Stream<G>>) -> Result<TensorMap<G>, G::Error> {
        let on_device = self
            .inputs
            .iter()
            .map(|(id, value)| Buffer::from_host(stream, value).map(|tensor| (id, tensor)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;

        let res = on_device
            .into_iter()
            .map(|(id, value)| value.value().map(|v| (id.clone(), v.0)))
            .collect::<Result<_, _>>()?;

        stream.sync()?;

        Ok(res)
    }
}
