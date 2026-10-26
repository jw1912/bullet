use std::collections::{BTreeMap, BTreeSet};

use bullet_compiler::{
    ir::NodeId,
    model::{Layout, MType, ModelBuilder, ModelIR, ModelNode},
    tensor::{
        DType, DValue, IRTrace, TValue, TensorIR,
        operation::{CABinary, Matmul, MatrixLayout, PadAcrossDimension},
        transform::{
            autograd::{LowerForward, TakeGradient},
            canonicalise::CanonicalisePass,
            inline::InlineSubgraphs,
        },
    },
};

use crate::model::{ModelInputs, inputs::InputType};

type NodeMap = BTreeMap<NodeId, NodeId>;

pub struct ModelFunctionDefinition {
    ir: TensorIR,
    map: NodeMap,
}

impl ModelFunctionDefinition {
    pub fn ir(&self) -> &TensorIR {
        &self.ir
    }

    pub fn map(&self) -> &NodeMap {
        &self.map
    }
}

pub struct ModelDefinition {
    ir: ModelIR,
    loss: Option<NodeId>,
    outputs: Vec<(NodeId, String)>,
    /// Enable validation of compiler passes in `TensorIR` compiled from this definition
    pub debug: bool,
}

pub struct GradNorms {
    /// Node with a sequence of gradient norms.
    pub acc: NodeId,
    /// Associated names of parameter groups.
    pub names: Vec<String>,
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

        Self { ir, loss, outputs: outputs.into(), debug: false }
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
        fwd.debug = self.debug;

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
        track_grad_norms: bool,
        record_optimisations: bool,
    ) -> Result<(ModelFunctionDefinition, NodeMap, Option<GradNorms>), IRTrace> {
        // if track_grad_norms is on,              ^^^^^^^^^^^^^^^^^
        // return a node with that in it,
        // and the associated param group names.

        let (mut bwd, map) = self.ir.lower(batch_size)?;
        bwd.debug = self.debug;

        let loss = self.loss.ok_or("Loss node not defined!")?;
        let loss = *map.get(&loss).unwrap();
        bwd.register_output(loss);

        if record_optimisations {
            bwd.start_recording("Optimise forward pass");
        }

        bwd.optimise()?;
        bwd.transform(CanonicalisePass::peephole_activations())?;
        bwd.stop_recording();

        let grad = bwd.add_const(TValue::F32(vec![1.0]));
        let op = bwd.get_parent_op(loss)?;
        let (transform, grads) = TakeGradient::new(op, [grad]);
        bwd.transform(transform)?;

        let mut gmap = BTreeMap::default();
        let mut sqnorms = Vec::new();

        for (&id, (name, _)) in self.ir.weights() {
            if !frozen.contains(&id) {
                let wid = *map.get(&id).unwrap();
                let ty = self.ir.node(id).ty();
                let size = ty.shape().size();
                let Layout::Dense(dtype) = ty.layout() else { unreachable!() };

                // weights with no path to the loss have zero gradient
                let gid =
                    grads.borrow().get(&wid).copied().unwrap_or_else(|| bwd.add_scalar(DValue::zero(dtype), size));

                if track_grad_norms {
                    let lhs = MatrixLayout { rows: 1.into(), cols: size.into(), col_mjr: true };
                    let rhs = MatrixLayout { rows: size.into(), cols: 1.into(), col_mjr: true };
                    sqnorms.push((name.clone(), bwd.add_op([gid, gid], Matmul::new(dtype, 1, lhs, rhs))?[0]));
                }

                bwd.register_output(gid);
                gmap.insert(id, gid);
            }
        }

        // cat squared norms into a single vector
        // [a, b, c] = [a, 0, 0]
        //           + [0, b, 0]
        //           + [0, 0, c]
        let norms = if sqnorms.is_empty() {
            None
        } else {
            let (names, sqnorms) = sqnorms.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
            let n = sqnorms.len();

            let mut cat = bwd.add_scalar(DValue::zero(DType::F32), n);
            for (i, sqnorm) in sqnorms.into_iter().enumerate() {
                let pad = PadAcrossDimension::new([1], 0, i, n - 1 - i, DValue::zero(DType::F32));
                let term = bwd.add_op([sqnorm], pad)?[0];
                cat = bwd.add_binary(cat, term, CABinary::Add)?;
            }

            bwd.register_output(cat);

            Some(GradNorms { acc: cat, names })
        };

        bwd.transform(LowerForward)?;
        bwd.transform(InlineSubgraphs)?;

        if record_optimisations {
            bwd.start_recording("Take gradient");
        }

        bwd.optimise()?;
        bwd.stop_recording();

        Ok((ModelFunctionDefinition { ir: bwd, map }, gmap, norms))
    }
}
