pub mod rng;
pub mod save;
pub mod utils;

use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::{
    ir::NodeId,
    model::{InitSettings, Layout, MType, ModelDefinition, ModelIR},
    tensor::{DType, TValue},
};
use bullet_gpu::{
    buffer::Buffer,
    function::Function,
    runtime::{Device, Gpu},
};

pub type TensorMap<G> = BTreeMap<String, Arc<Buffer<G>>>;

struct EvalutateModel<G: Gpu> {
    func: Function<G>,
    bufs: BTreeMap<NodeId, Arc<Buffer<G>>>,
    inputs: BTreeMap<String, NodeId>,
    outputs: BTreeMap<String, Arc<Buffer<G>>>,
}

pub struct Model<G: Gpu> {
    device: Arc<Device<G>>,
    weights: TensorMap<G>,
    definition: ModelDefinition,
    evaluate: EvalutateModel<G>,
}

impl<G: Gpu> Model<G> {
    pub fn new(
        ir: ModelIR,
        device: Arc<Device<G>>,
        loss: Option<NodeId>,
        outputs: impl Into<Vec<(NodeId, String)>>,
    ) -> Self {
        let mut weights = BTreeMap::new();

        for (&id, (name, init)) in ir.weights() {
            let node = ir.node(id);
            let size = node.ty().single_size();

            assert!(node.ty().is_dense() && !node.ty().is_batched());

            let init = match init {
                InitSettings::Zeroed => vec![0.0; size],
                InitSettings::Uniform { mean, stdev } => rng::vec_f32(size, *mean, *stdev, false),
                InitSettings::Normal { mean, stdev } => rng::vec_f32(size, *mean, *stdev, true),
                InitSettings::Custom(value) => value.f32().to_vec(),
            };

            let init = TValue::F32(init);
            let tensor = Buffer::from_host(&device, &init).unwrap();
            weights.insert(name.clone(), tensor);
        }

        if let Some(loss) = loss {
            assert_eq!(ir.node(loss).ty(), MType::new(false, 1, 1, Layout::Dense(DType::F32)));
        }

        let definition = ModelDefinition::new(ir.clone(), loss, outputs);
        let forward = definition.lower_forward(1).unwrap();

        let mut bufs = BTreeMap::new();
        for (id, (name, _)) in ir.weights() {
            let tid = *forward.map().get(id).unwrap();
            bufs.insert(tid, weights.get(name).unwrap().clone());
        }

        let mut inputs = BTreeMap::new();
        for (id, name) in ir.inputs() {
            let tid = *forward.map().get(id).unwrap();
            inputs.insert(name.clone(), tid);
        }

        let mut outputs = BTreeMap::new();
        for (id, name) in definition.outputs() {
            let tid = *forward.map().get(id).unwrap();
            let ty = forward.ir().get_node(tid).unwrap().ty();
            let buf = Buffer::zeroed(&device, ty.dtype(), ty.size().get()).unwrap();

            outputs.insert(name.clone(), buf.clone());
            bufs.insert(tid, buf);
        }

        let func = Function::new(device.clone(), forward.ir().clone()).unwrap();

        Model { weights, definition, evaluate: EvalutateModel { func, bufs, inputs, outputs }, device }
    }

    pub fn evaluate(&mut self, inputs: &TensorMap<G>) -> &TensorMap<G> {
        self.evaluate.func.prealloc().unwrap();

        for (input, buf) in inputs {
            if let Some(id) = self.evaluate.inputs.get(input) {
                self.evaluate.bufs.insert(*id, buf.clone());
            }
        }

        let stream = self.device.new_stream().unwrap();
        self.evaluate.func.execute(stream, &self.evaluate.bufs).unwrap().value().unwrap();
        &self.evaluate.outputs
    }

    pub fn device(&self) -> Arc<Device<G>> {
        self.device.clone()
    }

    pub fn weights(&self) -> &TensorMap<G> {
        &self.weights
    }

    pub fn definition(&self) -> &ModelDefinition {
        &self.definition
    }

    pub fn get_weights(&self, id: impl Into<String>) -> Option<TValue> {
        self.weights.get(&id.into()).cloned().map(|tensor| tensor.clone().to_host().unwrap())
    }

    pub fn set_weights(&self, id: impl Into<String>, new_value: &TValue) -> bool {
        self.weights.get(&id.into()).cloned().map(|tensor| tensor.copy_from_host(new_value).unwrap()).is_some()
    }

    pub fn write_weights_into(&self, writer: &mut impl std::io::Write) -> Result<(), G::Error> {
        let mut buf = Vec::new();

        for (id, value) in self.weights.clone() {
            if value.dtype() != DType::F32 {
                unimplemented!("Non f32 writing!");
            }

            let this_buf = value.clone().to_host()?;
            let byte_buf = utils::write_to_byte_buffer(&this_buf, &id).unwrap();
            buf.extend_from_slice(&byte_buf);
        }

        writer.write_all(&buf).unwrap();

        Ok(())
    }

    /// Loads the weights of a graph from a file. If `gradients` is true,
    /// it will instead load the gradients of those weights.
    pub fn load_weights_from(&mut self, mut reader: impl std::io::Read) -> Result<(), G::Error> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();

        let mut offset = 0;

        while offset < buf.len() {
            let (buffer, id, bytes_read) = utils::read_from_byte_buffer(&buf[offset..]);
            let weights = self.weights.get(&id).expect("No weight with ID found!").clone();

            if weights.dtype() != DType::F32 {
                unimplemented!("Non f32 writing!");
            }

            if buffer.len() != weights.size() {
                panic!("Invalid buffer size!");
            }

            weights.copy_from_host(&TValue::F32(buffer))?;

            offset += bytes_read;
        }

        Ok(())
    }
}
