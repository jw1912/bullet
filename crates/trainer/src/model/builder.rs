use std::{
    collections::{HashMap, HashSet},
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{Arc, Mutex, MutexGuard},
};

use bullet_compiler::{
    ir::{
        builder::IRNode,
        frontend::IRBuilder,
        graph::{DType, DValue, NodeId, Size, TType, TValue},
        operation::{
            BroadcastAcrossDimension, CABinary, CABinaryOp, Matmul, MatrixLayout, PadAcrossDimension,
            ReduceAcrossDimension, Reduction, SparseMatmul, Unary, UnaryOp,
        },
        transform::inline::InlineSubgraphs,
    },
    runtime::{Device, ReadyToCompileGraph, Stream, TensorInput},
};

use crate::model::{
    Model, Shape,
    autograd::{LowerForward, TakeGradient},
};

use super::autograd::{Autograd, AutogradOp};

#[derive(Clone, Copy, Debug)]
pub enum InitSettings {
    Zeroed,
    Normal { mean: f32, stdev: f32 },
    Uniform { mean: f32, stdev: f32 },
}

type InputDesc = (String, Shape, Option<usize>);

#[derive(Default)]
pub struct ModelBuilder {
    ir: IRBuilder,
    init: Mutex<HashMap<NodeId, InitSettings>>,
    inputs: Mutex<HashMap<NodeId, InputDesc>>,
}

impl ModelBuilder {
    fn init(&self) -> MutexGuard<'_, HashMap<NodeId, InitSettings>> {
        self.init.try_lock().unwrap()
    }

    fn inputs(&self) -> MutexGuard<'_, HashMap<NodeId, InputDesc>> {
        self.inputs.try_lock().unwrap()
    }

    pub fn add_op<'a>(&'a self, inputs: impl AsRef<[ModelNode<'a>]>, op: impl Autograd) -> Vec<NodeId> {
        let inputs = inputs.as_ref().iter().map(ModelNode::detach).collect::<Vec<_>>();
        let op = self.ir.add_op(inputs, AutogradOp::new(op).unwrap()).unwrap();
        op.iter().map(IRNode::node).collect()
    }

    pub fn scalar<'a>(&'a self, value: f32) -> ModelNode<'a> {
        let node = self.ir.scalar(DValue::F32(value), 1).node();
        let shape = Shape::new(1, 1);
        ModelNode { builder: self, node, shape, batched: false, sparse: None }
    }

    pub fn new_dense_input<'a>(&'a self, id: &str, shape: Shape) -> ModelNode<'a> {
        let node = self.ir.add_input(Size::variable() * shape.size(), DType::F32).node();
        assert!(self.inputs().insert(node, (format!("inputs/{id}"), shape, None)).is_none());
        ModelNode { node, builder: self, batched: true, shape, sparse: None }
    }

    pub fn new_sparse_input<'a>(&'a self, id: &str, shape: Shape, nnz: usize) -> ModelNode<'a> {
        let node = self.ir.add_input(Size::variable() * nnz, DType::I32).node();
        assert!(self.inputs().insert(node, (format!("inputs/{id}"), shape, Some(nnz))).is_none());
        ModelNode { node, builder: self, batched: true, shape, sparse: Some(nnz) }
    }

    pub fn new_constant<'a>(&'a self, shape: Shape, vals: &[f32]) -> ModelNode<'a> {
        assert_eq!(shape.size(), vals.len());
        let node = self.ir.constant(TValue::F32(vals.to_vec())).node();
        ModelNode { node, builder: self, batched: false, shape, sparse: None }
    }

    pub fn new_weights<'a>(&'a self, id: &str, shape: Shape, init: InitSettings) -> ModelNode<'a> {
        let node = self.ir.add_input(shape.size(), DType::F32).node();
        assert!(self.inputs().insert(node, (format!("weights/{id}"), shape, None)).is_none());
        self.init().insert(node, init);
        ModelNode { node, builder: self, batched: false, shape, sparse: None }
    }

    pub fn new_affine(&self, id: &str, input_size: usize, output_size: usize) -> Affine<'_> {
        self.new_affine_custom(id, input_size, output_size, 1)
    }

    pub fn new_affine_custom(&self, id: &str, input_size: usize, output_size: usize, bias_cols: usize) -> Affine<'_> {
        let wid = format!("{id}w");
        let init = InitSettings::Normal { mean: 0.0, stdev: (2.0 / (input_size as f32 * bias_cols as f32)).sqrt() };
        let weights = self.new_weights(&wid, Shape::new(output_size, input_size), init);
        let bias = self.new_weights(&format!("{id}b"), Shape::new(output_size, bias_cols), InitSettings::Zeroed);

        Affine { weights, bias }
    }

    pub fn build<'a, D: Device>(&'a self, device: Arc<D>, mut loss: ModelNode<'a>, output: ModelNode<'a>) -> Model<D> {
        assert_eq!(loss.shape, Shape::new(1, 1));

        if loss.batched {
            loss = loss.reduce_sum_across_batch();
        }

        let mut fwd = self.ir.build([output.detach()]);
        let fwd_ty = fwd.get_node(output.node).unwrap().ty();
        fwd.transform(LowerForward).unwrap();
        fwd.transform(InlineSubgraphs).unwrap();
        fwd.optimise().unwrap();

        let mut bwd = self.ir.build([loss.detach()]);
        let bwd_ty = bwd.get_node(loss.node).unwrap().ty();
        let grad = bwd.add_const(TValue::F32(vec![1.0]));
        let op = bwd.get_parent_op(loss.node).unwrap();
        let (transform, grads) = TakeGradient::new(op, [grad]);
        bwd.transform(transform).unwrap();
        bwd.transform(LowerForward).unwrap();
        bwd.transform(InlineSubgraphs).unwrap();

        let mut fwd_tensors = HashMap::new();
        let mut bwd_tensors = HashMap::new();
        let mut shapes = HashMap::new();
        let mut weights = HashMap::new();
        let stream = device.default_stream();

        let mut names = HashSet::new();

        for (id, (name, shape, sparse)) in self.inputs().clone() {
            assert!(names.insert(name.clone()));

            shapes.insert(name.clone(), (shape, sparse));
            bwd_tensors.insert(name.clone(), TensorInput::In(id));

            if fwd.get_node(id).is_ok() {
                fwd_tensors.insert(name.clone(), TensorInput::In(id));
            }

            if name.starts_with("weights/") {
                let tensor = stream.make_blocking(&TValue::F32(vec![0.0; shape.size()]));
                weights.insert(name.clone(), tensor.unwrap());

                let name = name.strip_prefix("weights/").unwrap().to_string();
                let gid = grads.borrow().get(&id).unwrap().unwrap();
                bwd.register_output(gid);
                bwd_tensors.insert(format!("gradients/{name}"), TensorInput::Out(gid));
            }
        }

        fwd_tensors.insert("outputs/output".to_string(), TensorInput::Out(output.node));
        let ready_fwd = ReadyToCompileGraph::new(fwd, fwd_tensors).unwrap();

        bwd_tensors.insert("outputs/loss".to_string(), TensorInput::Out(loss.node));
        bwd.optimise().unwrap();
        let ready_bwd = ReadyToCompileGraph::new(bwd, bwd_tensors).unwrap();

        Model {
            weights,
            shapes,
            forward: device.compile(ready_fwd).unwrap(),
            backward: device.compile(ready_bwd).unwrap(),
            fwd_output_types: [("outputs/output".to_string(), fwd_ty)].into(),
            bwd_output_types: [("outputs/loss".to_string(), bwd_ty)].into(),
            device,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Affine<'a> {
    pub weights: ModelNode<'a>,
    pub bias: ModelNode<'a>,
}

impl<'a> Affine<'a> {
    pub fn forward(self, input: ModelNode<'a>) -> ModelNode<'a> {
        self.weights.matmul(input) + self.bias
    }

    pub fn init_with_effective_input_size(&self, size: usize) {
        *self.weights.builder.init().get_mut(&self.weights.node).unwrap() =
            InitSettings::Normal { mean: 0.0, stdev: (2.0 / size as f32).sqrt() };
    }
}

#[derive(Clone, Copy)]
pub struct ModelNode<'a> {
    builder: &'a ModelBuilder,
    node: NodeId,
    shape: Shape,
    batched: bool,
    sparse: Option<usize>,
}

impl<'a> ModelNode<'a> {
    pub fn detach(&self) -> IRNode<'a> {
        IRNode::new(&self.builder.ir, self.node)
    }

    pub fn ty(&self) -> TType {
        self.detach().ty()
    }

    pub fn node(&self) -> NodeId {
        self.node
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn is_batched(&self) -> bool {
        self.batched
    }

    pub fn sparse(&self) -> Option<usize> {
        self.sparse
    }

    pub fn reshape(&self, shape: Shape) -> Self {
        if shape.size() != self.shape.size() {
            panic!("Mismatched shapes!");
        }

        Self { shape, ..*self }
    }

    pub fn broadcast_across_batch(self) -> Self {
        assert!(!self.batched);
        let dtype = self.ty().dtype();
        let size = self.shape.size();
        let op = BroadcastAcrossDimension::new(dtype, [size], 0, Size::variable());
        let node = self.builder.add_op([self], op.unwrap())[0];
        Self { node, batched: true, ..self }
    }

    pub fn reduce_sum_across_batch(self) -> Self {
        assert!(self.batched);
        let dtype = self.ty().dtype();
        let shape = [Size::variable(), self.shape.size().into()];
        let op = ReduceAcrossDimension::new(dtype, shape, 0, Reduction::Sum);
        let node = self.builder.add_op([self], op.unwrap())[0];
        Self { node, batched: false, ..self }
    }

    fn broadcast_scalar(self, shape: Shape) -> Self {
        assert!(!self.batched);
        assert_eq!(self.shape, Shape::new(1, 1));
        let dtype = self.ty().dtype();
        let op = BroadcastAcrossDimension::new(dtype, [1], 0, shape.size());
        let node = self.builder.add_op([self], op.unwrap())[0];
        Self { node, shape, ..self }
    }

    pub fn unary(self, unary: Unary) -> Self {
        let node = self.builder.add_op([self], UnaryOp::new(self.ty(), unary).unwrap())[0];
        Self { node, ..self }
    }

    pub fn binary(mut self, mut rhs: Self, binary: CABinary) -> Self {
        if self.shape.size() != rhs.shape.size() {
            if self.shape == Shape::new(1, 1) && !self.batched {
                self = self.broadcast_scalar(rhs.shape);
            }

            if rhs.shape == Shape::new(1, 1) && !rhs.batched {
                rhs = rhs.broadcast_scalar(self.shape);
            }
        }

        match (self.batched, rhs.batched) {
            (false, true) => self = self.broadcast_across_batch(),
            (true, false) => rhs = rhs.broadcast_across_batch(),
            _ => {}
        }

        let op = CABinaryOp::new(self.ty(), binary);
        let node = self.builder.add_op([self, rhs], op)[0];
        Self { node, ..self }
    }

    pub fn sparse_matmul(self, sparse: Self) -> Self {
        let Some(nnz) = sparse.sparse else { panic!("Node is not sparse!") };
        assert!(self.sparse.is_none());
        assert!(!self.batched);
        assert_eq!(self.shape.cols, sparse.shape.rows);
        assert_eq!(sparse.shape.cols, 1);

        let dtype = self.ty().dtype();
        let batch = if sparse.batched { Size::variable() } else { 1.into() };
        let shape = Shape::new(self.shape.rows, 1);
        let matmul = SparseMatmul::new(dtype, batch, self.shape.rows, self.shape.cols, nnz);
        let node = self.builder.add_op([self, sparse], matmul)[0];

        Self { node, batched: sparse.batched, shape, ..self }
    }

    pub fn matmul(self, other: Self) -> Self {
        if self.sparse.is_some() {
            unimplemented!("Sparse on left hand side of matmul!");
        }

        if other.sparse.is_some() {
            return self.sparse_matmul(other);
        }

        if self.ty().dtype() != other.ty().dtype() {
            panic!("Mismatched DTypes!");
        }

        let (batch, m, n, k) = match (self.batched, other.batched) {
            (false, false) => (1.into(), self.shape.rows, self.shape.cols, other.shape.cols.into()),
            (true, true) => (Size::variable(), self.shape.rows, self.shape.cols, other.shape.cols.into()),
            (false, true) => (1.into(), self.shape.rows, self.shape.cols, Size::variable()),
            (true, false) => unimplemented!(),
        };

        let lhs = MatrixLayout { rows: m.into(), cols: n.into(), col_mjr: true };
        let rhs = MatrixLayout { rows: n.into(), cols: k, col_mjr: true };
        let matmul = Matmul::new(self.ty().dtype(), batch, lhs, rhs).unwrap();

        let node = self.builder.add_op([self, other], matmul)[0];
        let shape = Shape::new(self.shape.rows, other.shape.cols);
        let batched = self.batched | other.batched;

        Self { builder: self.builder, node, shape, batched, sparse: None }
    }

    pub fn min(self, value: f32) -> Self {
        self.binary(self.builder.scalar(value), CABinary::Min)
    }

    pub fn max(self, value: f32) -> Self {
        self.binary(self.builder.scalar(value), CABinary::Max)
    }

    pub fn relu(self) -> Self {
        self.max(0.0)
    }

    pub fn crelu(self) -> Self {
        self.max(0.0).min(1.0)
    }

    pub fn screlu(self) -> Self {
        let x = self.crelu();
        x * x
    }

    pub fn exp(self) -> Self {
        self.unary(Unary::Exp)
    }

    pub fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }

    pub fn squared_error(self, other: Self) -> Self {
        let diff = self - other;
        diff * diff
    }

    pub fn pad(self, before: usize, after: usize, value: f32) -> Self {
        assert_eq!(self.shape.cols(), 1);
        let size = self.shape.size();

        let op = PadAcrossDimension::new(
            if self.batched { vec![Size::variable(), size.into()] } else { vec![size.into()] },
            usize::from(self.batched),
            before,
            after,
            value.into(),
        );

        let node = self.builder.add_op([self], op.unwrap())[0];

        Self { node, ..self }
    }

    pub fn concat(self, rhs: Self) -> Self {
        self.pad(0, rhs.shape.size(), 0.0) + rhs.pad(self.shape.size(), 0, 0.0)
    }

    pub fn pairwise_mul(self) -> Self {
        unimplemented!()
    }

    pub fn select(self, _indices: Self) -> Self {
        unimplemented!()
    }

    pub fn slice_rows(self, _start: usize, _end: usize) -> Self {
        unimplemented!()
    }

    pub fn repeat(self, _reps: usize) -> Self {
        unimplemented!()
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
