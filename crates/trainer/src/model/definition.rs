use std::collections::{BTreeMap, BTreeSet};

use bullet_compiler::{
    ir::NodeId,
    model::{Layout, MType, ModelBuilder, ModelIR, ModelNode},
    tensor::{
        DType, IRTrace, TValue, TensorIR,
        transform::{
            autograd::{LowerForward, TakeGradient},
            canonicalise::CanonicalisePass,
            inline::InlineSubgraphs,
        },
    },
};

use crate::model::{ModelInputs, inputs::InputType};

pub struct ModelFunctionDefinition {
    ir: TensorIR,
    map: BTreeMap<NodeId, NodeId>,
}

impl ModelFunctionDefinition {
    pub fn ir(&self) -> &TensorIR {
        &self.ir
    }

    pub fn map(&self) -> &BTreeMap<NodeId, NodeId> {
        &self.map
    }
}

pub struct ModelDefinition {
    ir: ModelIR,
    loss: Option<NodeId>,
    outputs: Vec<(NodeId, String)>,
}

impl ModelDefinition {
    pub fn build<T, F>(inputs: &ModelInputs<T>, f: F) -> Self
    where
        T: InputType,
        F: for<'a> FnOnce(&'a ModelBuilder, T::Nodes<'a>) -> (Option<ModelNode<'a>>, Vec<(String, ModelNode<'a>)>),
    {
        let builder = ModelBuilder::default();

        let nodes = inputs.make_nodes(&builder);
        let (loss, outputs) = f(&builder, nodes);
        let outputs = outputs.into_iter().map(|(name, node)| (node.node(), name)).collect::<Vec<_>>();

        Self::new(builder.inner(), loss.as_ref().map(ModelNode::node), outputs)
    }

    pub fn new(ir: ModelIR, loss: Option<NodeId>, outputs: impl Into<Vec<(NodeId, String)>>) -> Self {
        if let Some(loss) = loss {
            assert_eq!(ir.node(loss).ty(), MType::new(false, 1, 1, Layout::Dense(DType::F32)));
        }

        Self { ir, loss, outputs: outputs.into() }
    }

    pub fn ir(&self) -> &ModelIR {
        &self.ir
    }

    pub fn loss(&self) -> Option<NodeId> {
        self.loss
    }

    pub fn outputs(&self) -> &[(NodeId, String)] {
        &self.outputs
    }

    pub fn lower_forward(&self, batch_size: usize) -> Result<ModelFunctionDefinition, IRTrace> {
        let (mut fwd, map) = self.ir.lower(batch_size)?;

        for (output, _) in &self.outputs {
            fwd.register_output(*map.get(output).unwrap());
        }

        fwd.transform(LowerForward)?;
        fwd.transform(InlineSubgraphs)?;
        fwd.optimise()?;

        Ok(ModelFunctionDefinition { ir: fwd, map })
    }

    pub fn lower_backward(
        &self,
        frozen: &BTreeSet<NodeId>,
        batch_size: usize,
    ) -> Result<(ModelFunctionDefinition, BTreeMap<NodeId, NodeId>), IRTrace> {
        let (mut bwd, map) = self.ir.lower(batch_size)?;

        let loss = self.loss.ok_or("Loss node not defined!")?;
        let loss = *map.get(&loss).unwrap();
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
