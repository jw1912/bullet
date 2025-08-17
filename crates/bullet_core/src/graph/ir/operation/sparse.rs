use acyclib::graph::NodeId;

use crate::{
    function::{self, DeviceFunction},
    graph::{
        Graph, GraphNodeIdTy,
        ir::{
            BackendMarker, GraphIR, GraphIRError,
            node::AnnotatedNode,
            operation::{
                GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, unary::DiffableFromOutput,
                util,
            },
            shape::Shape,
        },
    },
};

#[derive(Clone, Debug)]
pub struct SparseAffineActivate {
    pub weights: AnnotatedNode,
    pub biases: Option<AnnotatedNode>,
    pub indices: AnnotatedNode,
    pub values: Option<AnnotatedNode>,
    pub activation: DiffableFromOutput,
}

impl<B: BackendMarker> GraphIROperationBase<B> for SparseAffineActivate {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        let mut nodes = vec![self.weights, self.indices];

        if let Some(v) = self.values {
            nodes.push(v);
        }

        if let Some(b) = self.biases {
            nodes.push(b);
        }

        nodes
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.weights, true)?;
        util::check_dense_eq(ir, &self.indices, false)?;
        util::check_not_batched(ir, &self.weights)?;
        util::check_no_grad(ir, &[&self.indices])?;

        if let Some(b) = &self.biases {
            util::check_dense_eq(ir, b, true)?;
        }

        let out = util::check_matmul(self.weights.shape, self.indices.shape)?;
        let mut check = self.biases.is_none() || out == self.biases.unwrap().shape;
        check &= self.indices.shape.cols() == 1;

        if let Some(v) = &self.values {
            util::check_dense_eq(ir, v, true)?;
            util::check_same_batching(ir, &[&self.indices, v])?;
            util::check_no_grad(ir, &[v])?;
            let nnz = ir.get(self.indices.idx).unwrap().ty().sparse.unwrap();
            check &= v.shape.cols() == 1 && v.shape.rows() == nnz.get();
        }

        check.then_some(out).ok_or(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.indices.shape)))
    }

    fn shorthand(&self) -> String {
        match (self.biases.is_some(), self.activation) {
            (true, DiffableFromOutput::Identity) => "SparseAffine".to_string(),
            (true, act) => format!("SparseAffine{act:?}"),
            (false, DiffableFromOutput::Identity) => "SparseMatmul".to_string(),
            (false, act) => format!("SparseMatmul{act:?}"),
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for SparseAffineActivate {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let indices = graph.get_ref(self.indices.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize { input: indices.clone(), output: output.clone() });

        func.push(function::SparseAffineActivate {
            weights: graph.get_ref(self.weights.idx, GraphNodeIdTy::Values),
            weights_shape: self.weights.shape,
            biases: self.biases.map(|b| graph.get_ref(b.idx, GraphNodeIdTy::Values)),
            input_shape: self.indices.shape,
            indices,
            values: self.values.map(|v| graph.get_ref(v.idx, GraphNodeIdTy::Values)),
            activation: self.activation,
            output,
        });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(weights_grads) = graph.maybe_get_ref(self.weights.idx, GraphNodeIdTy::Gradients) {
            let indices = graph.get_ref(self.indices.idx, GraphNodeIdTy::Values);

            if let Some(bias) = self.biases
                && let Some(output) = graph.maybe_get_ref(bias.idx, GraphNodeIdTy::Gradients)
                && output.borrow().batch_size().is_some()
            {
                func.push(function::MaybeUpdateBatchSize { input: indices.clone(), output });
            }

            func.push(function::BackpropSparseAffineActivate {
                weights_grads,
                weights_shape: self.weights.shape,
                biases_grads: self.biases.map(|b| graph.get_ref(b.idx, GraphNodeIdTy::Gradients)),
                input_shape: self.indices.shape,
                indices,
                values: self.values.map(|v| graph.get_ref(v.idx, GraphNodeIdTy::Values)),
                activation: self.activation,
                output: graph.get_ref(output_node, GraphNodeIdTy::Values),
                output_grads: graph.get_ref(output_node, GraphNodeIdTy::Gradients),
            });
        } else if let Some(b) = self.biases
            && graph.maybe_get_ref(b.idx, GraphNodeIdTy::Gradients).is_some()
        {
            todo!();
        }

        func
    }
}
