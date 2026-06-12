use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::ir::NodeId;
use bullet_gpu::{
    buffer::Buffer,
    function::Function,
    runtime::{Device, Gpu, Stream},
};

use crate::model::{ModelDefinition, ModelWeights, TensorMap};

pub struct ModelEvaluator<G: Gpu> {
    stream: Arc<Stream<G>>,
    func: Function<G>,
    bufs: BTreeMap<NodeId, Arc<Buffer<G>>>,
    weights: BTreeMap<String, NodeId>,
    inputs: BTreeMap<String, NodeId>,
    outputs: BTreeMap<String, Arc<Buffer<G>>>,
}

impl<G: Gpu> ModelEvaluator<G> {
    pub fn new(defn: &ModelDefinition, device: Arc<Device<G>>) -> Result<Self, G::Error> {
        let forward = defn.lower_forward(1).map_err(|e| format!("{e}"))?;

        let mut bufs = BTreeMap::new();

        let mut weights = BTreeMap::new();
        for (id, (name, _)) in defn.ir().weights() {
            let tid = *forward.map().get(id).unwrap();
            weights.insert(name.clone(), tid);
        }

        let mut inputs = BTreeMap::new();
        for (id, name) in defn.ir().inputs() {
            let tid = *forward.map().get(id).unwrap();
            inputs.insert(name.clone(), tid);
        }

        let mut outputs = BTreeMap::new();
        for (id, name) in defn.outputs() {
            let tid = *forward.map().get(id).unwrap();
            let ty = forward.ir().get_node(tid).map_err(|e| format!("{e}"))?.ty();
            let buf = Buffer::zeroed(&device, ty.dtype(), ty.size().get())?;

            outputs.insert(name.clone(), buf.clone());
            bufs.insert(tid, buf);
        }

        let stream = device.new_stream()?;
        let func = Function::new(device, forward.ir().clone()).map_err(|e| format!("{e}"))?;

        Ok(Self { stream, func, bufs, weights, inputs, outputs })
    }

    pub fn load_weights(&mut self, weights: &ModelWeights) -> Result<(), G::Error> {
        self.load_device_weights(&weights.to_device(&self.stream.device())?)
    }

    pub fn load_device_weights(&mut self, weights: &TensorMap<G>) -> Result<(), G::Error> {
        for (input, buf) in weights {
            if let Some(id) = self.weights.get(input) {
                self.bufs.insert(*id, buf.clone());
            }
        }

        Ok(())
    }

    pub fn evaluate(&mut self, inputs: &TensorMap<G>) -> Result<&TensorMap<G>, G::Error> {
        self.func.prealloc()?;

        for (input, buf) in inputs {
            if let Some(id) = self.inputs.get(input) {
                self.bufs.insert(*id, buf.clone());
            }
        }

        self.func.execute(self.stream.clone(), &self.bufs)?.value()?;
        Ok(&self.outputs)
    }
}
