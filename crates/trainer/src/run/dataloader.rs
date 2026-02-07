use std::{collections::HashMap, sync::Arc};

use bullet_compiler::{
    ir::graph::TValue,
    runtime::{Device, Stream},
};

use crate::model::TensorMap;

pub trait DataLoader: Send + Sync + 'static {
    type Error: Send + Sync;

    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(self, batch_size: usize, f: F) -> Result<(), Self::Error>;
}

pub struct PreparedBatchHost {
    pub batch_size: usize,
    pub inputs: HashMap<String, TValue>,
}

impl PreparedBatchHost {
    pub fn to_device_blocking<D: Device>(&self, stream: &Arc<D::Stream>) -> Result<TensorMap<D>, D::Error> {
        let on_device = self
            .inputs
            .iter()
            .map(|(id, value)| stream.clone().make_nonblocking(value).map(|tensor| (id, tensor)))
            .collect::<Result<HashMap<_, _>, _>>()?;

        let res = on_device.into_iter().map(|(id, value)| (id.clone(), value.value())).collect();

        stream.block_until_done()?;

        Ok(res)
    }
}
