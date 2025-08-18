use std::{collections::HashMap, sync::Arc};

use acyclib::manager::DAGraphManagerError;

use crate::{
    device::Device,
    function::{DeviceFunction, Set},
    graph::{
        Graph, GraphNodeId, GraphNodeIdTy,
        builder::Shape,
        ir::{
            BackendMarker, GraphIRError, GraphIRManager, GraphIRResult, GraphIRType,
            node::NodeInfo,
            passes::{self, GraphIRPass},
        },
    },
    tensor::{Tensor, TensorRef},
};

#[derive(Debug)]
pub struct GraphIRCompileError(pub String);

impl From<GraphIRError> for GraphIRCompileError {
    fn from(value: GraphIRError) -> Self {
        GraphIRCompileError(format!("{value:?}"))
    }
}

impl<B: BackendMarker> From<DAGraphManagerError<GraphIRType<B>>> for GraphIRCompileError {
    fn from(value: DAGraphManagerError<GraphIRType<B>>) -> Self {
        GraphIRCompileError(format!("{value:?}"))
    }
}

impl<B: BackendMarker> GraphIRManager<B>
where
    B::Backend: Device,
{
    pub fn optimise(&mut self) -> GraphIRResult<(), B> {
        // Elides concat:
        // Concat(PairwiseMul(a), PairwiseMul(b)) -> FusedPairwiseMulConcat(a, b)
        self.apply_pass(passes::FusePairwiseMulWithConcat)?;

        // Strict speedup by performing elementwise on far less values:
        // Select(Elementwise(op, [a, b, ..]), buckets) -> Elementwise(op, [Select(a, buckets), Select(b, buckets), ..])
        self.apply_pass(passes::ExchangeElementwiseAndSelect)?;

        // Looking to find a Matmul(c, Concat(a, b)):
        // Unary(Concat(a, b), op) -> Concat(Unary(a, op), Unary(b, op))
        self.apply_pass(passes::ExchangeConcatAndUnary)?;

        // Elides concat:
        // Matmul(c, Concat(a, b)) -> Add(Matmul(Slice(c, left half), a), Matmul(Slice(c, right half), b))
        self.apply_pass(passes::ExchangeMatmulAndConcatWithSliceAndMatmul)?;

        // Re-apply because above pass ends in an Add, which may be followed by a Select
        self.apply_pass(passes::ExchangeElementwiseAndSelect)?;

        // Add(SparseMatmul(..), b) -> SparseAffine(.., b)
        self.apply_pass(passes::FuseSparseMatmulWithAdd)?;

        // Unary(SparseAffine(..), DiffableFromOutput(activation)) -> SparseAffineActivate(.., activation)
        self.apply_pass(passes::FuseSparseAffineWithDiffableFromOutput)?;

        // Miscellaneous fusions that don't impact performance that much:
        // : AbsPow(Sub(a, b), power) -> AbsPowerErr(a, b, power)
        // : Mul(LinearCombination((a, a_wgt), ..), val) -> LinearCombination((a, val * a_wgt), ..)
        // : LinearCombination(.., (LinearCombination((a0, a0wgt), .., (aN, aNwgt)), wgt), ..)
        //       -> LinearCombination(.., (a0, wgt * a0wgt), .., (aN, wgt * aNwgt), ..)
        // : Add(Matmul(a, b), c) -> Affine(a, b, c)
        self.apply_pass(passes::LowPriorityFusions)?;

        Ok(())
    }

    pub fn apply_pass(&mut self, pass: impl GraphIRPass<B>) -> GraphIRResult<(), B> {
        loop {
            let roots = self.inner.roots();

            if self.inner.modify(|ir| pass.try_pass(ir).map_err(Into::into))? {
                self.inner.eliminate_dead_nodes(roots)?;
            } else {
                break;
            }
        }

        Ok(())
    }

    pub fn compile(self, device: B::Backend) -> Result<Graph<B::Backend>, GraphIRCompileError> {
        let root = self.root()?.idx;
        let root_data = self.get(root).unwrap().ty();

        if !root_data.requires_grad || root_data.batched || root_data.shape != Shape::new(1, 1) {
            return Err(GraphIRCompileError("Invalid root node in GraphIRManager".to_string()));
        }

        let device = Arc::new(device);

        let mut nodes = HashMap::new();

        let id_idx_pair = |node| self.ids.get(node).map(|data| (data.clone(), *node));
        let inputs = self.inputs.iter().filter_map(id_idx_pair).collect();
        let weights = self.weights.iter().filter_map(id_idx_pair).collect();

        let topo = self.inner.topo_order()?;

        let make_tensor = |device, shape, sparse, batched| {
            Tensor::new(device, shape, sparse, batched)
                .map_err(|_| GraphIRCompileError("Failed to initialise tensor!".to_string()))
                .map(TensorRef::new)
        };

        for ir_node in topo.iter().map(|&idx| self.get(idx).unwrap()) {
            let idx = ir_node.id();
            let op = ir_node.op();
            let NodeInfo { shape, sparse, requires_grad, batched } = ir_node.ty();

            let values = make_tensor(device.clone(), shape, sparse, batched)?;
            nodes.insert(GraphNodeId::new(idx, GraphNodeIdTy::Values), values);

            if requires_grad {
                let grads = make_tensor(device.clone(), shape, sparse, batched)?;
                let id = GraphNodeId::new(idx, GraphNodeIdTy::Gradients);
                nodes.insert(id, grads.clone());
            }

            let ancillary_buffers = op.ancillary_buffers(self.inner.current())?;

            for (num, &(shape, sparse, batched)) in ancillary_buffers.iter().enumerate() {
                let ancillary = make_tensor(device.clone(), shape, sparse, batched)?;
                let id = GraphNodeId::new(idx, GraphNodeIdTy::Ancillary(num as u16));
                nodes.insert(id, ancillary);
            }
        }

        let mut graph = Graph { nodes, inputs, weights, functions: HashMap::new(), device, root };

        let mut forward = DeviceFunction::default();
        let mut backward = DeviceFunction::default();
        let mut zero_grads = DeviceFunction::default();

        for ir_node in topo.into_iter().map(|idx| self.get(idx).unwrap()) {
            let idx = ir_node.id();
            let op = ir_node.op();
            let requires_grad = ir_node.ty().requires_grad;

            forward.extend(op.forward_pass(&graph, idx));

            if requires_grad {
                let id = graph.get_ref(idx, GraphNodeIdTy::Gradients);
                zero_grads.push(Set { id, val: 0.0 });

                let mut this_bwd = op.backward_pass(&graph, idx);
                this_bwd.extend(backward);
                backward = this_bwd;
            }
        }

        let mut new_bwd = DeviceFunction::default();
        let id = graph.get_ref(root, GraphNodeIdTy::Gradients);
        new_bwd.push(Set { id, val: 1.0 });
        new_bwd.extend(backward);
        backward = new_bwd;

        graph.functions.insert("forward".to_string(), forward);
        graph.functions.insert("backward".to_string(), backward);
        graph.functions.insert("zero_grads".to_string(), zero_grads);

        Ok(graph)
    }
}
