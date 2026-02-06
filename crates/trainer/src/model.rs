mod builder;

pub use builder::{ModelBuilder, ModelNode};

use std::collections::HashMap;

use bullet_compiler::graph::TValue;
use bullet_runtime::device::{BlockOnDrop, Device, Stream, TensorRef};

type TensorMap<D> = HashMap<String, TensorRef<<D as Device>::S>>;
type ModelError<D> = <<D as Device>::S as Stream>::Error;
type ModelAsyncReturn<D> = BlockOnDrop<<D as Device>::S, Vec<TensorRef<<D as Device>::S>>>;

pub struct Model<D: Device> {
    device: D,
    weights: TensorMap<D>,
    forward: <D::S as Stream>::CompiledGraph,
    backward: <D::S as Stream>::CompiledGraph,
}

impl<D: Device> Model<D> {
    pub fn get_weights(&self, id: impl Into<String>) -> Option<TValue> {
        self.weights
            .get(&id.into())
            .cloned()
            .map(|tensor| self.device.default_stream().copy_d2h_blocking(tensor).unwrap())
    }

    pub fn set_weights(&self, id: impl Into<String>, new_value: &TValue) -> bool {
        self.weights
            .get(&id.into())
            .cloned()
            .map(|tensor| self.device.default_stream().copy_h2d_blocking(new_value, tensor).unwrap())
            .is_some()
    }

    pub fn forward(
        &self,
        stream: D::S,
        inputs: TensorMap<D>,
        outputs: TensorMap<D>,
    ) -> Result<ModelAsyncReturn<D>, ModelError<D>> {
        let tensors = collect_map::<D>([("weights", &self.weights), ("inputs", &inputs), ("outputs", &outputs)]);

        stream.execute(&self.forward, tensors)
    }

    pub fn backward(
        &self,
        stream: D::S,
        inputs: TensorMap<D>,
        outputs: TensorMap<D>,
        gradients: TensorMap<D>,
    ) -> Result<ModelAsyncReturn<D>, ModelError<D>> {
        let tensors = collect_map::<D>([
            ("weights", &self.weights),
            ("inputs", &inputs),
            ("outputs", &outputs),
            ("gradients", &gradients),
        ]);

        stream.execute(&self.backward, tensors)
    }
}

fn collect_map<'a, D: Device + 'a>(x: impl AsRef<[(&'a str, &'a TensorMap<D>)]>) -> TensorMap<D> {
    let mut map = HashMap::new();

    for (pre, submap) in x.as_ref() {
        for (name, value) in submap.iter() {
            map.insert(format!("{pre}/{name}"), value.clone());
        }
    }

    map
}
