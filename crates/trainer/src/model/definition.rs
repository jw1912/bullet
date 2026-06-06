use std::collections::{BTreeMap, BTreeSet};

use bullet_compiler::{
    ir::NodeId,
    model::ModelIR,
    tensor::{
        IRTrace, TValue, TensorIR,
        transform::{
            autograd::{LowerForward, TakeGradient},
            canonicalise::CanonicalisePass,
            inline::InlineSubgraphs,
        },
    },
};

pub struct ModelFunctionDefinition {
    pub ir: TensorIR,
    pub map: BTreeMap<NodeId, NodeId>,
}

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

    pub fn lower_forward(&self, batch_size: usize) -> Result<ModelFunctionDefinition, IRTrace> {
        let (mut fwd, map) = self.ir.lower(batch_size)?;

        for output in &self.outputs {
            fwd.register_output(*map.get(output).unwrap());
        }

        fwd.optimise()?;

        Ok(ModelFunctionDefinition { ir: fwd, map })
    }

    pub fn lower_backward(
        &self,
        frozen: &BTreeSet<NodeId>,
        batch_size: usize,
    ) -> Result<(ModelFunctionDefinition, BTreeMap<NodeId, NodeId>), IRTrace> {
        let (mut bwd, map) = self.ir.lower(batch_size)?;

        let loss = *map.get(&self.loss).unwrap();
        bwd.register_output(loss);
        bwd.optimise()?;
        bwd.transform(CanonicalisePass::peephole_activations())?;

        let grad = bwd.add_const(TValue::F32(vec![1.0]));
        let op = bwd.get_parent_op(loss)?;
        let (transform, grads) = TakeGradient::new(op, [grad]);
        bwd.transform(transform)?;

        let mut gmap = BTreeMap::default();

        for &id in self.ir.weights().keys() {
            if !frozen.contains(&id) {
                let wid = *map.get(&id).unwrap();
                let gid = *grads.borrow().get(&wid).unwrap();
                bwd.register_output(gid);
                gmap.insert(id, gid);
            }
        }

        bwd.transform(LowerForward)?;
        bwd.transform(InlineSubgraphs)?;
        bwd.optimise()?;

        Ok((ModelFunctionDefinition { ir: bwd, map }, gmap))
    }
}
