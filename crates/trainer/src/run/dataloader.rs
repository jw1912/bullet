use std::{collections::HashMap, sync::Arc};

use bullet_compiler::graph::TValue;

use crate::{
    model::TensorMap,
    runtime::{Device, Stream},
};

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
    pub inputs: HashMap<String, TValue>,
}

impl PreparedBatchHost {
    pub fn to_device_blocking<S: Stream>(
        self,
        stream: &Arc<S>,
    ) -> Result<TensorMap<S::Device>, <S::Device as Device>::Error> {
        let on_device = self
            .inputs
            .iter()
            .map(|(id, value)| stream.clone().make_nonblocking(value).map(|tensor| (id, tensor)))
            .collect::<Result<HashMap<_, _>, _>>()?;

        let res = on_device.into_iter().map(|(id, value)| (id.clone(), value.value().1.clone())).collect();

        stream.block_until_done()?;

        Ok(res)
    }
}
