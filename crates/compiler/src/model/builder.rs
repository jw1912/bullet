use std::{
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{Mutex, MutexGuard},
};

use crate::{
    ir::NodeId,
    model::{Layout, MType, ModelOperation, ModelIR, operations::*},
    tensor::{
        DType, DValue, TValue,
        operation::{CABinary, Reduction, Unary},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
}

impl Shape {
    pub fn new(rows: usize, cols: usize) -> Self {
        Shape { rows, cols }
    }

    pub fn size(&self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Clone, Copy, Debug)]
pub enum InitSettings {
    Zeroed,
    Normal { mean: f32, stdev: f32 },
    Uniform { mean: f32, stdev: f32 },
}

#[derive(Default)]
pub struct ModelBuilder {
    ir: Mutex<ModelIR>,
}

impl ModelBuilder {
    fn ir(&self) -> MutexGuard<ModelIR> {
        self.ir.try_lock().unwrap()
    }

    pub fn add_op<'a>(&'a self, inputs: impl AsRef<[ModelNode<'a>]>, op: impl ModelOperation) -> NodeId {
        let inputs = inputs.as_ref().iter().map(ModelNode::node).collect::<Vec<_>>();
        self.ir().add_op(inputs, op).unwrap()
    }

    pub fn scalar<'a>(&'a self, value: f32) -> ModelNode<'a> {
        let value = TValue::from(DValue::from(value));
        ModelNode { builder: self, node: self.add_op([], Constant::new(value, 1, 1)) }
    }

    pub fn new_dense_input<'a>(&'a self, shape: Shape) -> ModelNode<'a> {
        ModelNode { builder: self, node: self.ir().add_input(true, shape.rows, shape.cols, Layout::Dense(DType::F32)) }
    }

    pub fn new_sparse_input<'a>(&'a self, shape: Shape, nnz: usize) -> ModelNode<'a> {
        ModelNode { builder: self, node: self.ir().add_input(true, shape.rows, shape.cols, Layout::Sparse(nnz)) }
    }

    pub fn new_constant<'a>(&'a self, shape: Shape, vals: &[f32]) -> ModelNode<'a> {
        assert_eq!(shape.size(), vals.len());
        let value = TValue::F32(vals.to_vec());
        ModelNode { builder: self, node: self.add_op([], Constant::new(value, shape.rows, shape.cols)) }
    }

    pub fn new_weights<'a>(&'a self, shape: Shape) -> ModelNode<'a> {
        ModelNode { builder: self, node: self.ir().add_input(false, shape.rows, shape.cols, Layout::Dense(DType::F32)) }
    }
}

#[derive(Clone, Copy)]
pub struct ModelNode<'a> {
    builder: &'a ModelBuilder,
    node: NodeId,
}

impl<'a> ModelNode<'a> {
    pub fn node(&self) -> NodeId {
        self.node
    }

    pub fn ty(&self) -> MType {
        self.builder.ir().ir.node(self.node).unwrap().ty()
    }

    pub fn reshape(self, shape: Shape) -> Self {
        Self { node: self.builder.add_op([self], Reshape::new(self.ty(), shape.rows, shape.cols)), ..self }
    }

    pub fn broadcast_across_batch(self) -> Self {
        let broadcast = Broadcast(self.ty(), Dim::Batch, None);
        Self { node: self.builder.add_op([self], broadcast), ..self }
    }

    pub fn reduce_sum_across_batch(self) -> Self {
        let reduce = Reduce(self.ty(), Dim::Batch, Reduction::Sum);
        Self { node: self.builder.add_op([self], reduce), ..self }
    }

    fn broadcast_scalar(self, rows: usize, cols: usize) -> Self {
        let broadcast = Broadcast(self.ty(), Dim::Rows, Some(rows * cols));
        Self { node: self.builder.add_op([self], broadcast), ..self }.reshape(Shape { rows, cols })
    }

    pub fn unary(self, unary: Unary) -> Self {
        Self { node: self.builder.add_op([self], PointwiseUnary(self.ty(), unary)), ..self }
    }

    pub fn binary(mut self, mut rhs: Self, binary: CABinary) -> Self {
        let sty = self.ty();
        let rty = rhs.ty();

        if sty.single_size() != rty.single_size() {
            if sty.single_size() == 1 && !sty.batch {
                self = self.broadcast_scalar(rty.rows, rty.cols);
            }

            if rty.single_size() == 1 && !rty.batch {
                rhs = rhs.broadcast_scalar(sty.rows, sty.cols);
            }
        }

        match (sty.batch, rty.batch) {
            (false, true) => self = self.broadcast_across_batch(),
            (true, false) => rhs = rhs.broadcast_across_batch(),
            _ => {}
        }

        Self { node: self.builder.add_op([self, rhs], PointwiseBinary(self.ty(), binary)), ..self }
    }

    /*pub fn sparse_matmul(self, sparse: Self) -> Self {
        let Some(nnz) = sparse.nt.sparse else { panic!("Node is not sparse!") };
        assert!(self.nt.sparse.is_none());
        assert!(!self.nt.batched);
        assert_eq!(self.nt.shape.cols, sparse.nt.shape.rows);
        assert_eq!(sparse.nt.shape.cols, 1);

        let dtype = self.ty().dtype();
        let batch = if sparse.nt.batched { Size::variable() } else { 1.into() };
        let shape = Shape::new(self.nt.shape.rows, 1);
        let matmul = SparseMatmul::new(dtype, batch, self.nt.shape.rows, self.nt.shape.cols, nnz);
        let node = self.builder.add_op([self, sparse], matmul)[0];

        Self { node, nt: NodeType { batched: sparse.nt.batched, shape, ..self.nt }, ..self }
    }*/

    pub fn matmul(self, other: Self) -> Self {
        let lty = self.ty();
        let rty = other.ty();

        let ldtype = match lty.layout {
            Layout::Sparse(_) => unimplemented!("Sparse on left hand side of matmul!"),
            Layout::Dense(dtype) => dtype,
        };

        let rdtype = match rty.layout {
            Layout::Sparse(_) => return self.sparse_matmul(other),
            Layout::Dense(dtype) => dtype,
        };

        if ldtype != rdtype {
            panic!("Mismatched DTypes!");
        }

        if lty.cols != rty.rows {
            panic!("Mismatched shapes: {} != {}", lty.cols, rty.rows);
        }

        let matmul =
            Matmul { lbatch: lty.batch, rbatch: rty.batch, m: lty.rows, n: lty.cols, k: rty.rows, dtype: ldtype };
        Self { node: self.builder.add_op([self, other], matmul), ..self }
    }

    pub fn min(self, value: f32) -> Self {
        self.binary(self.builder.scalar(value), CABinary::Min)
    }

    pub fn max(self, value: f32) -> Self {
        self.binary(self.builder.scalar(value), CABinary::Max)
    }

    pub fn relu(self) -> Self {
        Self { node: self.builder.add_op([self], ReLU(self.ty())), ..self }
    }

    pub fn crelu(self) -> Self {
        Self { node: self.builder.add_op([self], CReLU(self.ty())), ..self }
    }

    pub fn screlu(self) -> Self {
        Self { node: self.builder.add_op([self], SCReLU(self.ty())), ..self }
    }

    pub fn sigmoid(self) -> Self {
        Self { node: self.builder.add_op([self], Sigmoid(self.ty())), ..self }
    }

    pub fn exp(self) -> Self {
        self.unary(Unary::Exp)
    }

    pub fn abs(self) -> Self {
        self.unary(Unary::Abs)
    }

    pub fn abs_pow(self, power: f32) -> Self {
        (power * self.abs().unary(Unary::Log)).exp()
    }

    pub fn squared_error(self, other: Self) -> Self {
        let diff = self - other;
        diff * diff
    }

    pub fn faux_quantise(self, value: f32, round: bool) -> Self {
        let op = FauxQuantise(self.ty(), value.into(), round);
        Self { node: self.builder.add_op([self], op), ..self }
    }

    pub fn concat(self, _rhs: Self) -> Self {
        unimplemented!()
    }

    pub fn pairwise_mul(self) -> Self {
        let size = self.ty().rows;
        self.slice_rows(0, size / 2) * self.slice_rows(size / 2, size)
    }

    pub fn select(mut self, mut indices: Self) -> Self {
        let sty = self.ty();
        let ity = indices.ty();

        assert_eq!(ity.layout, Layout::Sparse(1));
        assert_eq!(sty.cols, 1);

        let Layout::Dense(dtype) = sty.layout else { panic!() };

        let batched = sty.batch | ity.batch;
        match (sty.batch, ity.batch) {
            (false, true) => self = self.broadcast_across_batch(),
            (true, false) => indices = indices.broadcast_across_batch(),
            _ => {}
        }

        let inner = self.nt.shape.size();
        let divisor = indices.shape().size();

        let op = Select { dtype, batch, inner, divisor };
        let rows = inner / divisor;
        let node = self.builder.add_op([self, indices], op)[0];
        Self { builder: self.builder, node }
    }

    pub fn slice_rows(self, start: usize, end: usize) -> Self {
        let op = Slice::new(self.ty(), start, end, true);
        Self { node: self.builder.add_op([self], op), ..self }
    }

    pub fn repeat(self, reps: usize) -> Self {
        let broadcast = Broadcast(self.ty(), Dim::Cols, Some(reps));
        Self { node: self.builder.add_op([self], broadcast), ..self }
    }
}

impl Add<Self> for ModelNode<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, CABinary::Add)
    }
}

impl Sub<Self> for ModelNode<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<'a> Add<ModelNode<'a>> for f32 {
    type Output = ModelNode<'a>;

    fn add(self, rhs: ModelNode<'a>) -> Self::Output {
        rhs.builder.scalar(self) + rhs
    }
}

impl Add<f32> for ModelNode<'_> {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        rhs + self
    }
}

impl Neg for ModelNode<'_> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -1.0 * self
    }
}

impl<'a> Sub<ModelNode<'a>> for f32 {
    type Output = ModelNode<'a>;

    fn sub(self, rhs: ModelNode<'a>) -> Self::Output {
        self + (-1.0 * rhs)
    }
}

impl Sub<f32> for ModelNode<'_> {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        self + (-rhs)
    }
}

impl<'a> Mul<ModelNode<'a>> for f32 {
    type Output = ModelNode<'a>;

    fn mul(self, rhs: ModelNode<'a>) -> Self::Output {
        rhs.builder.scalar(self) * rhs
    }
}

impl<'a> Mul<ModelNode<'a>> for ModelNode<'a> {
    type Output = ModelNode<'a>;

    fn mul(self, rhs: ModelNode<'a>) -> Self::Output {
        self.binary(rhs, CABinary::Mul)
    }
}

impl Mul<f32> for ModelNode<'_> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        rhs * self
    }
}

impl Div<f32> for ModelNode<'_> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        (1.0 / rhs) * self
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> Div<ModelNode<'a>> for f32 {
    type Output = ModelNode<'a>;

    fn div(self, rhs: ModelNode<'a>) -> Self::Output {
        self * rhs.unary(Unary::Reciprocal)
    }
}
