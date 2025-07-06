use crate::graph::ir::{
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

#[derive(Debug)]
pub struct SparseAffineDualActivate {
    pub weights: AnnotatedNode,
    pub biases: Option<AnnotatedNode>,
    pub indices_l: AnnotatedNode,
    pub indices_r: AnnotatedNode,
    pub activation: DiffableFromOutput,
}

impl GraphIROperation for SparseAffineDualActivate {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        let mut nodes = vec![self.weights, self.indices_l, self.indices_r];

        if let Some(b) = self.biases {
            nodes.push(b);
        }

        nodes
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.weights, true)?;
        util::check_dense_eq(ir, &self.indices_l, false)?;
        util::check_dense_eq(ir, &self.indices_r, false)?;
        util::check_not_batched(ir, &self.weights)?;
        util::check_same_batching(ir, &[&self.indices_l, &self.indices_r])?;
        util::check_no_grad(ir, &[&self.indices_l, &self.indices_r])?;

        let out = util::check_matmul(self.weights.shape, self.indices_l.shape)?;
        let mut valid = self.indices_l.shape == self.indices_r.shape;

        if let Some(b) = self.biases {
            util::check_dense_eq(ir, &b, true)?;
            valid &= out == b.shape
        }

        valid.then_some(Shape::new(2 * out.rows(), out.cols())).ok_or(GraphIRError::Op(
            GraphIROperationError::MismatchedInputShapes(vec![
                self.weights.shape,
                self.indices_l.shape,
                self.indices_r.shape,
            ]),
        ))
    }
}
