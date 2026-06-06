use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bullet_compiler::{
    ir::NodeId,
    model::ModelIR,
    tensor::{
        TValue,
        transform::{
            autograd::{LowerForward, TakeGradient},
            canonicalise::CanonicalisePass,
            inline::InlineSubgraphs,
        },
    },
};

use bullet_gpu::{
    function::Function,
    runtime::{Device, Gpu},
};

pub struct ModelDefinition {
    ir: ModelIR,
    loss: NodeId,
    outputs: Vec<NodeId>,
}

impl ModelDefinition {
    pub fn new(ir: ModelIR, loss: NodeId, outputs: impl Into<Vec<NodeId>>) -> Self {
        Self { ir, loss, outputs: outputs.into() }
    }

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
