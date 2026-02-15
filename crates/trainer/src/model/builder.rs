use std::{
    collections::{HashMap, HashSet},
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use bullet_compiler::{
    frontend::{DType, DValue, IRBuilder, IRNode, Size, TType, TValue},
    graph::NodeId,
    operation::{
        BroadcastAcrossDimension, CABinary, CABinaryOp, Matmul, MatrixLayout, PadAcrossDimension,
        ReduceAcrossDimension, Reduction, Select, SliceAcrossDimension, SparseMatmul, Unary, UnaryOp,
    },
    transform::inline::InlineSubgraphs,
};

use crate::{
    model::{
        Model, Shape,
        autograd::{
            CReLU, DiffableFromOutput, DiffableFromOutputOp, LowerForward, ReLU, SCReLU, Sigmoid, TakeGradient,
        },
    },
    runtime::{Device, ReadyToCompileGraph, Stream, TensorInput},
};

use super::autograd::{Autograd, AutogradOp};

#[derive(Clone, Copy, Debug)]
pub enum InitSettings {
    Zeroed,
    Normal { mean: f32, stdev: f32 },
    Uniform { mean: f32, stdev: f32 },
}

type InputDesc = (String, Shape, Option<usize>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeType {
    pub shape: Shape,
    pub batched: bool,
    pub sparse: Option<usize>,
}

impl From<NodeType> for bullet_compiler::graph::Shape {
    fn from(value: NodeType) -> Self {
        let Shape { rows, cols } = value.shape;

        match (value.batched, value.sparse) {
            (true, None) => [Size::variable(), cols.into(), rows.into()].into(),
            (false, None) => [cols, rows].into(),
            (true, Some(nnz)) => [Size::variable(), nnz.into()].into(),
            (false, Some(nnz)) => [nnz].into(),
        }
    }
}

#[derive(Default)]
pub struct ModelBuilder {
    ir: IRBuilder,
    init: Mutex<HashMap<NodeId, InitSettings>>,
    inputs: Mutex<HashMap<NodeId, InputDesc>>,
    no_grad: AtomicBool,
    frozen: Mutex<HashSet<NodeId>>,
}

impl ModelBuilder {
    fn init(&self) -> MutexGuard<'_, HashMap<NodeId, InitSettings>> {
        self.init.try_lock().unwrap()
    }

    fn inputs(&self) -> MutexGuard<'_, HashMap<NodeId, InputDesc>> {
        self.inputs.try_lock().unwrap()
    }

    fn is_no_grad(&self) -> bool {
        self.no_grad.load(Ordering::SeqCst)
    }

    pub fn no_grad<T>(&self, mut f: impl FnMut() -> T) -> T {
        self.no_grad.store(true, Ordering::SeqCst);
        let ret = f();
        self.no_grad.store(false, Ordering::SeqCst);
        ret
    }

    pub fn add_op<'a>(&'a self, inputs: impl AsRef<[ModelNode<'a>]>, op: impl Autograd) -> Vec<NodeId> {
        let inputs = inputs.as_ref().iter().map(ModelNode::detach).collect::<Vec<_>>();
        let op = AutogradOp::new(op).unwrap();

        let outputs = if self.is_no_grad() {
            self.ir.add_op(inputs, op.into_forward()).unwrap()
        } else {
            self.ir.add_op(inputs, op).unwrap()
        };

        outputs.iter().map(IRNode::node).collect()
    }

    pub fn scalar<'a>(&'a self, value: f32) -> ModelNode<'a> {
        let node = self.ir.scalar(DValue::F32(value), 1).node();
        let shape = Shape::new(1, 1);
        ModelNode { builder: self, node, nt: NodeType { shape, batched: false, sparse: None } }
    }

    pub fn new_dense_input<'a>(&'a self, id: &str, shape: Shape) -> ModelNode<'a> {
        let node = self.ir.add_input(Size::variable() * shape.size(), DType::F32).node();
        assert!(self.inputs().insert(node, (format!("inputs/{id}"), shape, None)).is_none());
        ModelNode { node, builder: self, nt: NodeType { batched: true, shape, sparse: None } }
    }

    pub fn new_sparse_input<'a>(&'a self, id: &str, shape: Shape, nnz: usize) -> ModelNode<'a> {
        let node = self.ir.add_input(Size::variable() * nnz, DType::I32).node();
        assert!(self.inputs().insert(node, (format!("inputs/{id}"), shape, Some(nnz))).is_none());
        ModelNode { node, builder: self, nt: NodeType { batched: true, shape, sparse: Some(nnz) } }
    }

    pub fn new_constant<'a>(&'a self, shape: Shape, vals: &[f32]) -> ModelNode<'a> {
        assert_eq!(shape.size(), vals.len());
        let node = self.ir.constant(TValue::F32(vals.to_vec())).node();
        ModelNode { node, builder: self, nt: NodeType { batched: false, shape, sparse: None } }
    }

    pub fn new_weights<'a>(&'a self, id: &str, shape: Shape, init: InitSettings) -> ModelNode<'a> {
        let node = self.ir.add_input(shape.size(), DType::F32).node();
        assert!(self.inputs().insert(node, (format!("weights/{id}"), shape, None)).is_none());
        self.init().insert(node, init);

        if self.is_no_grad() {
            self.frozen.try_lock().unwrap().insert(node);
        }

        ModelNode { node, builder: self, nt: NodeType { batched: false, shape, sparse: None } }
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
        assert_eq!(loss.nt.shape, Shape::new(1, 1));

        if loss.nt.batched {
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

        let mut fwd_tensors = HashMap::new();
        let mut bwd_tensors = HashMap::new();
        let mut shapes = HashMap::new();
        let mut weights = HashMap::new();
        let stream = device.default_stream();

        let frozen = self.frozen.try_lock().unwrap().clone();
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

                if !frozen.contains(&id) {
                    bwd.register_output(gid);
                    bwd_tensors.insert(format!("gradients/{name}"), TensorInput::Out(gid));
                }
            }
        }

        fwd_tensors.insert("outputs/output".to_string(), TensorInput::Out(output.node));
        let ready_fwd = ReadyToCompileGraph::new(fwd, fwd_tensors).unwrap();

        bwd_tensors.insert("outputs/loss".to_string(), TensorInput::Out(loss.node));
        bwd.transform(LowerForward).unwrap();
        bwd.transform(InlineSubgraphs).unwrap();
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
    nt: NodeType,
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

    pub fn nt(&self) -> NodeType {
        self.nt
    }

    pub fn shape(&self) -> Shape {
        self.nt.shape
    }

    pub fn is_batched(&self) -> bool {
        self.nt.batched
    }

    pub fn sparse(&self) -> Option<usize> {
        self.nt.sparse
    }

    pub fn reshape(mut self, shape: Shape) -> Self {
        if shape.size() != self.nt.shape.size() {
            panic!("Mismatched shapes!");
        }

        self.nt.shape = shape;
        self
    }

    pub fn broadcast_across_batch(self) -> Self {
        assert!(!self.nt.batched);
        let dtype = self.ty().dtype();
        let size = self.shape().size();
        let op = BroadcastAcrossDimension::new(dtype, [size], 0, Size::variable());
        let node = self.builder.add_op([self], op.unwrap())[0];
        Self { node, nt: NodeType { batched: true, ..self.nt }, ..self }
    }

    pub fn reduce_sum_across_batch(self) -> Self {
        assert!(self.nt.batched);
        let dtype = self.ty().dtype();
        let shape = [Size::variable(), self.nt.shape.size().into()];
        let op = ReduceAcrossDimension::new(dtype, shape, 0, Reduction::Sum);
        let node = self.builder.add_op([self], op.unwrap())[0];
        Self { node, nt: NodeType { batched: false, ..self.nt }, ..self }
    }

    fn broadcast_scalar(self, shape: Shape) -> Self {
        assert!(!self.nt.batched);
        assert_eq!(self.nt.shape, Shape::new(1, 1));
        let dtype = self.ty().dtype();
        let op = BroadcastAcrossDimension::new(dtype, [1], 0, shape.size());
        let node = self.builder.add_op([self], op.unwrap())[0];
        Self { node, nt: NodeType { shape, ..self.nt }, ..self }
    }

    pub fn unary(self, unary: Unary) -> Self {
        let node = self.builder.add_op([self], UnaryOp::new(self.ty(), unary).unwrap())[0];
        Self { node, ..self }
    }

    pub fn binary(mut self, mut rhs: Self, binary: CABinary) -> Self {
        if self.nt.shape.size() != rhs.nt.shape.size() {
            if self.nt.shape == Shape::new(1, 1) && !self.nt.batched {
                self = self.broadcast_scalar(rhs.nt.shape);
            }

            if rhs.nt.shape == Shape::new(1, 1) && !rhs.nt.batched {
                rhs = rhs.broadcast_scalar(self.nt.shape);
            }
        }

        match (self.nt.batched, rhs.nt.batched) {
            (false, true) => self = self.broadcast_across_batch(),
            (true, false) => rhs = rhs.broadcast_across_batch(),
            _ => {}
        }

        let op = CABinaryOp::new(self.ty(), binary);
        let node = self.builder.add_op([self, rhs], op)[0];
        Self { node, ..self }
    }

    pub fn sparse_matmul(self, sparse: Self) -> Self {
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
    }

    pub fn matmul(self, other: Self) -> Self {
        if self.nt.sparse.is_some() {
            unimplemented!("Sparse on left hand side of matmul!");
        }

        if other.nt.sparse.is_some() {
            return self.sparse_matmul(other);
        }

        if self.ty().dtype() != other.ty().dtype() {
            panic!("Mismatched DTypes!");
        }

        let Shape { rows, cols } = self.nt.shape;

        let (batch, m, n, k) = match (self.nt.batched, other.nt.batched) {
            (false, false) => (1.into(), rows, cols, other.nt.shape.cols.into()),
            (true, true) => (Size::variable(), rows, cols, other.nt.shape.cols.into()),
            (false, true) => (1.into(), rows, cols, Size::variable()),
            (true, false) => unimplemented!(),
        };

        let lhs = MatrixLayout { rows: m.into(), cols: n.into(), col_mjr: true };
        let rhs = MatrixLayout { rows: n.into(), cols: k, col_mjr: true };
        let matmul = Matmul::new(self.ty().dtype(), batch, lhs, rhs).unwrap();

        let node = self.builder.add_op([self, other], matmul)[0];
        let shape = Shape::new(rows, other.nt.shape.cols);
        let batched = self.nt.batched | other.nt.batched;

        Self { builder: self.builder, node, nt: NodeType { shape, batched, sparse: None } }
    }

    pub fn min(self, value: f32) -> Self {
        self.binary(self.builder.scalar(value), CABinary::Min)
    }

    pub fn max(self, value: f32) -> Self {
        self.binary(self.builder.scalar(value), CABinary::Max)
    }

    fn dfo(self, op: impl DiffableFromOutput) -> Self {
        let op = DiffableFromOutputOp(op, self.ty().dtype(), self.ty().size());
        let node = self.builder.add_op([self], op)[0];
        Self { node, ..self }
    }

    pub fn relu(self) -> Self {
        self.dfo(ReLU)
    }

    pub fn crelu(self) -> Self {
        self.dfo(CReLU)
    }

    pub fn screlu(self) -> Self {
        self.dfo(SCReLU)
    }

    pub fn sigmoid(self) -> Self {
        self.dfo(Sigmoid)
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

    pub fn pad(self, before: usize, after: usize, value: f32) -> Self {
        let op = PadAcrossDimension::new(self.nt, 1 + usize::from(self.nt.batched), before, after, value.into());

        let node = self.builder.add_op([self], op.unwrap())[0];

        let shape = Shape::new(before + after + self.nt.shape.rows, self.nt.shape.cols);
        Self { node, nt: NodeType { shape, ..self.nt }, ..self }
    }

    pub fn concat(self, rhs: Self) -> Self {
        self.pad(0, rhs.nt.shape.rows, 0.0) + rhs.pad(self.nt.shape.rows, 0, 0.0)
    }

    pub fn pairwise_mul(self) -> Self {
        let size = self.nt.shape.rows;
        self.slice_rows(0, size / 2) * self.slice_rows(size / 2, size)
    }

    pub fn select(mut self, mut indices: Self) -> Self {
        assert!(self.nt.sparse.is_none());
        assert_eq!(indices.nt.sparse, Some(1));
        assert_eq!(self.nt.shape.cols, 1);

        let batched = self.nt.batched | indices.nt.batched;
        match (self.nt.batched, indices.nt.batched) {
            (false, true) => self = self.broadcast_across_batch(),
            (true, false) => indices = indices.broadcast_across_batch(),
            _ => {}
        }

        let dtype = self.ty().dtype();
        let batch = if batched { Size::variable() } else { 1.into() };
        let inner = self.nt.shape.size().into();
        let divisor = indices.shape().size().into();

        let op = Select { dtype, batch, inner, divisor };
        let rows = (inner / divisor).evaluate_constant().unwrap();
        let node = self.builder.add_op([self, indices], op)[0];
        Self { node, nt: NodeType { shape: Shape::new(rows, 1), batched, sparse: None }, ..self }
    }

    pub fn slice_rows(self, start: usize, end: usize) -> Self {
        let op = SliceAcrossDimension::new(self.ty().dtype(), self.nt, 1 + usize::from(self.nt.batched), start, end);

        let node = self.builder.add_op([self], op.unwrap())[0];

        let shape = Shape::new(end - start, self.nt.shape.cols);
        Self { node, nt: NodeType { shape, ..self.nt }, ..self }
    }

    pub fn repeat(self, reps: usize) -> Self {
        let op = BroadcastAcrossDimension::new(self.ty().dtype(), self.nt, usize::from(self.nt.batched), reps);

        let node = self.builder.add_op([self], op.unwrap())[0];

        let shape = Shape::new(self.nt.shape.rows, reps * self.nt.shape.cols);
        Self { node, nt: NodeType { shape, ..self.nt }, ..self }
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
