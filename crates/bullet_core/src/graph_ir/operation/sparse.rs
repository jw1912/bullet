use crate::graph_ir::{
    node::AnnotatedNode,
    operation::{unary::DiffableFromOutput, util, GraphIROperation, GraphIROperationError},
    shape::Shape,
    GraphIR, GraphIRError,
};

#[derive(Debug)]
pub struct SparseAffineActivate {
    pub weights: AnnotatedNode,
    pub biases: Option<AnnotatedNode>,
    pub indices: AnnotatedNode,
    pub values: Option<AnnotatedNode>,
    pub activation: DiffableFromOutput,
}

impl GraphIROperation for SparseAffineActivate {
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

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
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
            let nnz = ir.get(self.indices.idx).unwrap().sparse.unwrap();
            check &= v.shape.cols() == 1 && v.shape.rows() == nnz.get();
        }

        check.then_some(out).ok_or(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.indices.shape)))
    }
}
