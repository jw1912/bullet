use std::{
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{Mutex, MutexGuard},
};

use crate::{
    ir::NodeId,
    model::{InitSettings, Layout, MType, ModelIR, ModelOperation, operations::*},
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

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl From<(usize, usize)> for Shape {
    fn from((rows, cols): (usize, usize)) -> Self {
        Shape { rows, cols }
    }
}

#[derive(Default)]
pub struct ModelBuilder {
    ir: Mutex<ModelIR>,
}

impl ModelBuilder {
    pub fn ir(&'_ self) -> MutexGuard<'_, ModelIR> {
        self.ir.try_lock().unwrap()
    }

    pub fn inner(&self) -> ModelIR {
        self.ir().clone()
    }

    pub fn add_op<'a>(&'a self, inputs: impl AsRef<[ModelNode<'a>]>, op: impl ModelOperation) -> NodeId {
        let inputs = inputs.as_ref().iter().map(ModelNode::node).collect::<Vec<_>>();
        self.ir().add_op(inputs, op).unwrap()
    }

    pub fn scalar<'a>(&'a self, value: f32) -> ModelNode<'a> {
        let value = TValue::from(DValue::from(value));
        ModelNode { builder: self, node: self.add_op([], Constant::new(value, 1, 1)) }
    }

    pub fn new_dense_input<'a>(&'a self, name: impl Into<String>, shape: impl Into<Shape>) -> ModelNode<'a> {
        let shape = shape.into();
        ModelNode {
            builder: self,
            node: self.ir().add_input(name, true, shape.rows, shape.cols, Layout::Dense(DType::F32)),
        }
    }

    pub fn new_sparse_input<'a>(
        &'a self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        nnz: usize,
    ) -> ModelNode<'a> {
        let shape = shape.into();
        ModelNode { builder: self, node: self.ir().add_input(name, true, shape.rows, shape.cols, Layout::Sparse(nnz)) }
    }

    pub fn new_constant<'a>(&'a self, shape: impl Into<Shape>, vals: &[f32]) -> ModelNode<'a> {
        let shape = shape.into();
        assert_eq!(shape.size(), vals.len());
        let value = TValue::F32(vals.to_vec());
        ModelNode { builder: self, node: self.add_op([], Constant::new(value, shape.rows, shape.cols)) }
    }

    pub fn new_weights<'a>(
        &'a self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        init: InitSettings,
    ) -> ModelNode<'a> {
        let shape = shape.into();
        ModelNode { builder: self, node: self.ir().add_weight(name, shape.rows, shape.cols, init) }
    }

    pub fn new_affine(&self, id: &str, input_size: usize, output_size: usize) -> Affine<'_> {
        self.new_affine_custom(id, input_size, output_size, 1)
    }

    pub fn new_affine_custom(
        &self,
        id: impl AsRef<str>,
        input_size: usize,
        output_size: usize,
        bias_cols: usize,
    ) -> Affine<'_> {
        let init = InitSettings::Normal { mean: 0.0, stdev: (2.0 / (input_size as f32 * bias_cols as f32)).sqrt() };
        let weights = self.new_weights(format!("{}w", id.as_ref()), (output_size, input_size), init);
        let bias = self.new_weights(format!("{}b", id.as_ref()), (output_size, bias_cols), InitSettings::Zeroed);
        Affine { weights, bias }
    }

    pub fn with_no_grad<T>(&self, mut f: impl FnMut() -> T) -> T {
        let value = self.ir().stop_grad;
        self.ir().stop_grad = true;
        let out = f();
        self.ir().stop_grad = value;
        out
    }
}

#[derive(Clone, Copy)]
pub struct Affine<'a> {
    pub weights: ModelNode<'a>,
    pub bias: ModelNode<'a>,
}

impl<'a> Affine<'a> {
    /// Slice affine layer from `inputs -> outputs` to `inputs -> (end - start)`, so we have
    /// ```#
    /// affine.slice(start, end).forward(inputs) == affine.forward(inputs).slice_rows(start, end)
    /// ```
    pub fn slice(self, start: usize, end: usize) -> Self {
        Self { weights: self.weights.slice_rows(start, end), bias: self.bias.slice_rows(start, end) }
    }

    pub fn forward(self, input: ModelNode<'a>) -> ModelNode<'a> {
        self.weights.matmul(input) + self.bias
    }

    pub fn init_with_effective_input_size(&self, size: usize) {
        self.weights.builder.ir().weights.get_mut(&self.weights.node).unwrap().1 =
            InitSettings::Normal { mean: 0.0, stdev: (2.0 / size as f32).sqrt() };
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

    pub fn shape(&self) -> Shape {
        let ty = self.ty();
        Shape::new(ty.rows, ty.cols)
    }

    pub fn reshape(self, shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        Self { node: self.builder.add_op([self], Reshape::new(self.ty(), shape.rows, shape.cols)), ..self }
    }

    pub fn broadcast_across_batch(self) -> Self {
        let broadcast = Broadcast(self.ty(), Dim::Batch, None);
        Self { node: self.builder.add_op([self], broadcast), ..self }
    }

    #[deprecated(note = "Use `reduce_sum_batch` instead!")]
    pub fn reduce_sum_across_batch(self) -> Self {
        self.reduce_sum_batch()
    }

    pub fn reduce_sum_batch(self) -> Self {
        let reduce = Reduce(self.ty(), Dim::Batch, Reduction::Sum);
        Self { node: self.builder.add_op([self], reduce), ..self }
    }

    pub fn reduce_sum_rows(self) -> Self {
        let reduce = Reduce(self.ty(), Dim::Rows, Reduction::Sum);
        Self { node: self.builder.add_op([self], reduce), ..self }
    }

    pub fn reduce_sum_cols(self) -> Self {
        let reduce = Reduce(self.ty(), Dim::Cols, Reduction::Sum);
        Self { node: self.builder.add_op([self], reduce), ..self }
    }

    pub fn scalar_like(self, value: impl Into<DValue>) -> Self {
        let ty = self.ty();
        let value = match value.into() {
            DValue::F32(val) => TValue::F32(vec![val; ty.single_size()]),
            DValue::I32(val) => TValue::I32(vec![val; ty.single_size()]),
        };
        let node = self.builder.add_op([], Constant::new(value, ty.rows, ty.cols));
        let scalar = ModelNode { node, ..self };

        if ty.batch { scalar.broadcast_across_batch() } else { scalar }
    }

    fn broadcast_scalar(self, rows: usize, cols: usize) -> Self {
        let broadcast = Broadcast(self.ty(), Dim::Rows, Some(rows * cols));
        Self { node: self.builder.add_op([self], broadcast), ..self }.reshape((rows, cols))
    }

    fn broadcast_to_same(mut self, mut rhs: Self) -> (Self, Self) {
        let sty = self.ty();
        let rty = rhs.ty();

        if sty.single_size() != rty.single_size() {
            if sty.rows == 1 && sty.cols == 1 && !sty.batch {
                self = self.broadcast_scalar(rty.rows, rty.cols);
            }

            if rty.rows == 1 && rty.cols == 1 && !rty.batch {
                rhs = rhs.broadcast_scalar(sty.rows, sty.cols);
            }
        }

        match (sty.batch, rty.batch) {
            (false, true) => self = self.broadcast_across_batch(),
            (true, false) => rhs = rhs.broadcast_across_batch(),
            _ => {}
        }

        (self, rhs)
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

    pub fn sparse_matmul(self, sparse: Self) -> Self {
        let dsty = self.ty();
        let spty = sparse.ty();

        let Layout::Dense(dtype) = dsty.layout else { panic!("Node is not dense!") };
        let Layout::Sparse(nnz) = spty.layout else { panic!("Node is not sparse!") };
        assert_eq!(dsty.cols, spty.rows);
        assert_eq!(spty.cols, 1);

        let matmul = SparseMatmul { dtype, batch: spty.batch, rows: dsty.rows, cols: dsty.cols, nnz };

        Self { node: self.builder.add_op([self, sparse], matmul), ..self }
    }

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

        if ldtype != rdtype || lty.cols != rty.rows {
            panic!("Matmul: {lty} @ {rty} is not possible!");
        }

        let matmul =
            Matmul { lbatch: lty.batch, rbatch: rty.batch, m: lty.rows, n: lty.cols, k: rty.cols, dtype: ldtype };
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

    pub fn sqrrelu(self) -> Self {
        Self { node: self.builder.add_op([self], SqrReLU(self.ty())), ..self }
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

    pub fn abs_pow(mut self, power: f32) -> Self {
        let mut power = self.builder.scalar(power);
        (self, power) = self.broadcast_to_same(power);
        Self { node: self.builder.add_op([self, power], AbsPower(self.ty())), ..self }
    }

    pub fn squared_error(self, other: Self) -> Self {
        let diff = self - other;
        diff * diff
    }

    pub fn power_error(self, targets: Self, power: f32) -> Self {
        (self - targets).abs_pow(power)
    }

    pub fn softmax_crossentropy_loss(self, targets: Self) -> Self {
        let op = SoftmaxCrossEntropy(self.ty());
        Self { node: self.builder.add_op([self, targets], op), ..self }
    }

    pub fn clip_pass_through_grad(self, min: f32, max: f32) -> Self {
        let op = ClipPassThroughGrad(self.ty(), min, max);
        Self { node: self.builder.add_op([self], op), ..self }
    }

    pub fn faux_quantise(self, value: f32, round: bool) -> Self {
        let op = FauxQuantise(self.ty(), value.into(), round);
        Self { node: self.builder.add_op([self], op), ..self }
    }

    pub fn concat(self, rhs: Self) -> Self {
        let sty = self.ty();
        let rty = rhs.ty();

        let Layout::Dense(dtype) = sty.layout else { panic!() };

        assert_eq!(sty.layout, rty.layout);
        assert_eq!(sty.cols, rty.cols);
        assert_eq!(sty.cols, 1);
        assert_eq!(sty.batch, rty.batch);

        let op = Concat::new(dtype, sty.rows, rty.rows, sty.batch);
        Self { node: self.builder.add_op([self, rhs], op), ..self }
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
        assert_eq!(ity.cols, 1);

        let Layout::Dense(dtype) = sty.layout else { panic!() };

        let batched = sty.batch | ity.batch;
        match (sty.batch, ity.batch) {
            (false, true) => self = self.broadcast_across_batch(),
            (true, false) => indices = indices.broadcast_across_batch(),
            _ => {}
        }

        let op = SelectRows::new(dtype, sty.rows, ity.rows, batched);
        Self { node: self.builder.add_op([self, indices], op), ..self }
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
