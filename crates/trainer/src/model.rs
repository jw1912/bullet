pub mod rng;
pub mod save;
pub mod utils;

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bullet_compiler::{
    ir::NodeId,
    model::{InitSettings, Layout, MType, ModelIR},
    tensor::{
        DType, TValue,
        transform::{
            autograd::{LowerForward, TakeGradient},
            canonicalise::CanonicalisePass,
            inline::InlineSubgraphs,
        },
    },
};
use bullet_gpu::{
    buffer::Buffer,
    function::Function,
    runtime::{Device, Gpu},
};

pub type TensorMap<G> = BTreeMap<String, Arc<Buffer<G>>>;

pub struct ModelDefinition {
    ir: ModelIR,
    loss: NodeId,
    outputs: Vec<NodeId>,
}

impl ModelDefinition {
    pub fn ir(&self) -> &ModelIR {
        &self.ir
    }

    pub fn loss(&self) -> NodeId {
        self.loss
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub fn compile_backward<G: Gpu>(
        &self,
        frozen: &BTreeSet<NodeId>,
        batch_size: usize,
        device: Arc<Device<G>>,
    ) -> (Function<G>, BTreeMap<NodeId, NodeId>, BTreeMap<NodeId, NodeId>) {
        let (mut bwd, map) = self.ir.lower(batch_size).unwrap();

        let loss = *map.get(&self.loss).unwrap();
        bwd.register_output(loss);
        bwd.optimise().unwrap();
        bwd.transform(CanonicalisePass::peephole_activations()).unwrap();

        let grad = bwd.add_const(TValue::F32(vec![1.0]));
        let op = bwd.get_parent_op(loss).unwrap();
        let (transform, grads) = TakeGradient::new(op, [grad]);
        bwd.transform(transform).unwrap();

        let mut gmap = BTreeMap::default();

        for &id in self.ir.weights().keys() {
            if !frozen.contains(&id) {
                let wid = *map.get(&id).unwrap();
                let gid = *grads.borrow().get(&wid).unwrap();
                bwd.register_output(gid);
                gmap.insert(id, gid);
            }
        }

        bwd.transform(LowerForward).unwrap();
        bwd.transform(InlineSubgraphs).unwrap();
        bwd.optimise().unwrap();

        (Function::new(device, bwd).unwrap(), map, gmap)
    }
}

pub struct Model<G: Gpu> {
    device: Arc<Device<G>>,
    weights: TensorMap<G>,
    definition: ModelDefinition,
}

impl<G: Gpu> Model<G> {
    pub fn new(ir: &ModelIR, device: Arc<Device<G>>, loss: NodeId, outputs: impl Into<Vec<NodeId>>) -> Self {
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

        assert_eq!(ir.node(loss).ty(), MType::new(false, 1, 1, Layout::Dense(DType::F32)));

        Model { weights, device, definition: ModelDefinition { ir: ir.clone(), loss, outputs: outputs.into() } }
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
