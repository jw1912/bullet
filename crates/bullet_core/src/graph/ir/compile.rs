use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::{
    device::Device,
    graph::{
        builder::Shape,
        instruction::Set,
        ir::{
            node::{GraphIRNode, NodeInfo},
            passes::{self, GraphIRPass},
            BackendMarker, GraphIR, GraphIRError, GraphIRNodeInfo,
        },
        tensor::Tensor,
        Graph, GraphFunction, NodeId, NodeIdTy,
    },
};

#[derive(Debug)]
pub struct GraphIRCompileError(pub String);

impl From<GraphIRError> for GraphIRCompileError {
    fn from(value: GraphIRError) -> Self {
        GraphIRCompileError(format!("{value:?}"))
    }
}

impl<B: BackendMarker> GraphIR<B>
where
    B::Backend: Device,
{
    pub fn optimise(&mut self) -> Result<(), GraphIRError> {
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

    pub fn apply_pass(&mut self, pass: impl GraphIRPass<B>) -> Result<(), GraphIRError> {
        while let Some(transform) = pass.try_pass(self)? {
            self.apply_transform(transform)?;
        }

        Ok(())
    }

    pub fn compile(mut self, device: B::Backend) -> Result<Graph<B::Backend>, GraphIRCompileError> {
        if let Err(e) = self.check_valid() {
            return Err(GraphIRCompileError(format!("Compilation given invalid GraphIR: {e:?}")));
        }

        if let Some(path) = self.opts.dump_graphviz.clone() {
            use std::io::Write;
            let opts = "style=filled;\ncolor=lightgrey;\nnode [style=filled,color=white];\n";
            let unoptim = self.as_graphviz("unoptim").unwrap();
            let unoptim = format!("subgraph cluster_0 {{\nlabel=\"Unoptimised\";\n{opts}{unoptim}}}");

            if let Err(e) = self.optimise() {
                return Err(GraphIRCompileError(format!("Error encountered in optimising GraphIR: {e:?}")));
            }

            let optim = self.as_graphviz("optim").unwrap();
            let optim = format!("subgraph cluster_1 {{\nlabel=\"Optimised\";\n{opts}{optim}}}");

            let mut file = std::fs::File::create(path).unwrap();
            write!(&mut file, "digraph G {{\n{unoptim}\n{optim}}}").unwrap();
        } else if let Err(e) = self.optimise() {
            return Err(GraphIRCompileError(format!("Error encountered in optimising GraphIR: {e:?}")));
        }

        if let Err(e) = self.check_valid() {
            return Err(GraphIRCompileError(format!("Optimisation resulted in invalid GraphIR: {e:?}")));
        }

        let root = self.root()?.idx;
        let root_data = self.get(root).unwrap().info;

        if !root_data.requires_grad || root_data.batched || root_data.shape != Shape::new(1, 1) {
            return Err(GraphIRCompileError("Invalid root node in GraphIR".to_string()));
        }

        // populate ancillary buffers
        let mut ancillary_buffers = HashMap::new();

        for node in self.nodes.iter() {
            if let Some(op) = &node.1.parent_operation {
                ancillary_buffers.insert(node.0, op.ancillary_buffers(&self)?);
            }
        }

        let device = Arc::new(device);

        let mut nodes = HashMap::new();
        let mut forward = GraphFunction::default();
        let mut backward = GraphFunction::default();

        let mut zero_grads = GraphFunction::default();

        let id_idx_pair = |&node| self.get(node).ok().map(|data| (data.id.clone().unwrap(), node));
        let inputs = self.inputs.iter().filter_map(id_idx_pair).collect();
        let weights = self.weights.iter().filter_map(id_idx_pair).collect();

        let node_info =
            GraphIRNodeInfo { nodes: self.nodes.iter().map(|(idx, GraphIRNode { info, .. })| (*idx, *info)).collect() };

        let topo = self.topo_order()?;

        let make_tensor = |device, shape, sparse| {
            Tensor::new(device, shape, sparse)
                .map_err(|_| GraphIRCompileError("Failed to initialsie tensor!".to_string()))
        };

        for GraphIRNode { idx, info, parent_operation, .. } in topo.into_iter().map(|idx| self.get(idx).unwrap()) {
            let idx = *idx;
            let NodeInfo { shape, sparse, requires_grad, .. } = *info;

            let values = make_tensor(device.clone(), shape, sparse)?;
            nodes.insert(NodeId::new(idx, NodeIdTy::Values), RefCell::new(values));

            if requires_grad {
                let grads = make_tensor(device.clone(), shape, sparse)?;
                let id = NodeId::new(idx, NodeIdTy::Gradients);
                nodes.insert(id, RefCell::new(grads));

                zero_grads.push(Set { id, val: 0.0 });
            }

            if let Some(op) = parent_operation {
                for (num, &(shape, sparse)) in ancillary_buffers.get(&idx).unwrap().iter().enumerate() {
                    let ancillary = make_tensor(device.clone(), shape, sparse)?;
                    let id = NodeId::new(idx, NodeIdTy::Ancillary(num as u16));
                    nodes.insert(id, RefCell::new(ancillary));
                }

                forward.extend(op.forward_pass(&node_info, idx));

                if requires_grad {
                    let mut this_bwd = op.backward_pass(&node_info, idx);
                    this_bwd.extend(backward);
                    backward = this_bwd;
                }
            }
        }

        let mut new_bwd = GraphFunction::default();
        new_bwd.push(Set { id: NodeId { id: root, ty: NodeIdTy::Gradients }, val: 1.0 });
        new_bwd.extend(backward);
        backward = new_bwd;

        let functions = [("forward", forward), ("backward", backward), ("zero_grads", zero_grads)]
            .into_iter()
            .map(|(x, y)| (x.to_string(), y))
            .collect();

        Ok(Graph { nodes, inputs, weights, functions, device, root, profiles: HashMap::new() })
    }
}
