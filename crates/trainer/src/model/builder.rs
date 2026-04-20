use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use bullet_compiler::{
    ir::NodeId,
    tensor::{
        DType, DValue, IRBuilder, Size, TNode, TType, TValue,
        operation::{
            BroadcastAcrossDimension, CABinary, CABinaryOp, Matmul, MatrixLayout, PadAcrossDimension, PassThrough,
            Power, ReduceAcrossDimension, Reduction, Select, SliceAcrossDimension, SparseMatmul, Unary, UnaryOp,
            autograd::{
                Autograd, AutogradOp, CReLU, DiffableFromOutput, DiffableFromOutputOp, FauxQuantise, ReLU, SCReLU,
                Sigmoid, SoftmaxCrossEntropyLoss,
            },
        },
        transform::{
            autograd::{LowerForward, TakeGradient},
            inline::InlineSubgraphs,
        },
    },
};
use bullet_gpu::{
    buffer::Buffer,
    function::Function,
    runtime::{Device, Gpu},
};

use crate::model::{Model, Shape, rng};

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

impl From<NodeType> for bullet_compiler::tensor::Shape {
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
    init: Mutex<BTreeMap<NodeId, InitSettings>>,
    inputs: Mutex<BTreeMap<NodeId, InputDesc>>,
    no_grad: AtomicBool,
    frozen: Mutex<BTreeSet<NodeId>>,
}

impl ModelBuilder {
    fn init(&self) -> MutexGuard<'_, BTreeMap<NodeId, InitSettings>> {
        self.init.try_lock().unwrap()
    }

    fn inputs(&self) -> MutexGuard<'_, BTreeMap<NodeId, InputDesc>> {
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

        outputs.iter().map(TNode::node).collect()
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

    pub fn build<'a, G: Gpu>(
        &'a self,
        device: Arc<Device<G>>,
        mut loss: ModelNode<'a>,
        output: ModelNode<'a>,
    ) -> Model<G> {
        assert_eq!(loss.nt.shape, Shape::new(1, 1));

        if loss.nt.batched {
            loss = loss.reduce_sum_batch();
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

        let mut fwd_map = BTreeMap::new();
        let mut bwd_map = BTreeMap::new();
        let mut shapes = BTreeMap::new();
        let mut weights = BTreeMap::new();

        let frozen = self.frozen.try_lock().unwrap().clone();
        let mut names = BTreeSet::new();

        for (id, (name, shape, sparse)) in self.inputs().clone() {
            assert!(names.insert(name.clone()));

            shapes.insert(name.clone(), (shape, sparse));
            bwd_map.insert(name.clone(), id);

            if fwd.get_node(id).is_ok() {
                fwd_map.insert(name.clone(), id);
            }

            if name.starts_with("weights/") {
                let init = match self.init.lock().unwrap().get(&id).unwrap() {
                    InitSettings::Zeroed => vec![0.0; shape.size()],
                    InitSettings::Uniform { mean, stdev } => rng::vec_f32(shape.size(), *mean, *stdev, false),
                    InitSettings::Normal { mean, stdev } => rng::vec_f32(shape.size(), *mean, *stdev, true),
                };
                let init = TValue::F32(init);
                let tensor = Buffer::from_host(&device, &init).unwrap();
                let name = name.strip_prefix("weights/").unwrap().to_string();

                weights.insert(name.clone(), tensor);
                let gid = grads.borrow().get(&id).unwrap().unwrap();

                if !frozen.contains(&id) {
                    bwd.register_output(gid);
                    bwd_map.insert(format!("gradients/{name}"), gid);
                }
            }
        }

        fwd_map.insert("outputs/output".to_string(), output.node);
        bwd_map.insert("outputs/loss".to_string(), loss.node);
        bwd.transform(LowerForward).unwrap();
        bwd.transform(InlineSubgraphs).unwrap();
        bwd.optimise().unwrap();

        Model {
            weights,
            shapes,
            forward: Function::new(device.clone(), fwd).unwrap(),
            fwd_map,
            backward: Function::new(device.clone(), bwd).unwrap(),
            bwd_map,
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
    pub fn detach(&self) -> TNode<'a> {
        TNode::new(&self.builder.ir, self.node)
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

    #[deprecated(note = "Use `reduce_sum_batch` instead!")]
    pub fn reduce_sum_across_batch(self) -> Self {
        self.reduce_sum_batch()
    }

    pub fn reduce_sum_batch(self) -> Self {
        assert!(self.nt.batched);
        let dtype = self.ty().dtype();
        let shape = [Size::variable(), self.nt.shape.size().into()];
        let op = ReduceAcrossDimension::new(dtype, shape, 0, Reduction::Sum);
        let node = self.builder.add_op([self], op.unwrap())[0];
        Self { node, nt: NodeType { batched: false, ..self.nt }, ..self }
    }

    pub fn reduce_sum_rows(self) -> Self {
        let dtype = self.ty().dtype();
        let cols = self.nt.shape.cols;
        let op = ReduceAcrossDimension::new(dtype, self.nt, 1 + usize::from(self.is_batched()), Reduction::Sum);
        let node = self.builder.add_op([self], op.unwrap())[0];
        Self { node, nt: NodeType { shape: Shape { rows: 1, cols }, ..self.nt }, ..self }
    }

    pub fn reduce_sum_cols(self) -> Self {
        let dtype = self.ty().dtype();
        let rows = self.nt.shape.rows;
        let op = ReduceAcrossDimension::new(dtype, self.nt, usize::from(self.is_batched()), Reduction::Sum);
        let node = self.builder.add_op([self], op.unwrap())[0];
        Self { node, nt: NodeType { shape: Shape { rows, cols: 1 }, ..self.nt }, ..self }
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

    fn broadcast_to_same(mut self, mut rhs: Self) -> (Self, Self) {
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

        (self, rhs)
    }

    pub fn binary(mut self, mut rhs: Self, binary: CABinary) -> Self {
        (self, rhs) = self.broadcast_to_same(rhs);

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

    pub fn abs_pow(mut self, power: f32) -> Self {
        let abs = self.abs();
        let mut power = self.builder.scalar(power);
        (self, power) = self.broadcast_to_same(power);
        let node = self.builder.add_op([abs, power], Power(self.ty().size()))[0];
        Self { node, ..self }
    }

    pub fn squared_error(self, other: Self) -> Self {
        let diff = self - other;
        diff * diff
    }

    pub fn power_error(self, targets: Self, power: f32) -> Self {
        (self - targets).abs_pow(power)
    }

    pub fn faux_quantise(self, value: f32, round: bool) -> Self {
        let op = FauxQuantise(self.ty(), value.into(), round);
        let node = self.builder.add_op([self], op)[0];
        Self { node, ..self }
    }

    pub fn softmax_crossentropy_loss(self, targets: Self) -> Self {
        assert_eq!(self.is_batched(), targets.is_batched());

        let op = SoftmaxCrossEntropyLoss {
            batch_size: if self.is_batched() { Size::variable() } else { 1.into() },
            axis_size: self.shape().size(),
        };
        let node = self.builder.add_op([self, targets], op)[0];
        Self { node, ..self }
    }

    pub fn clip_pass_through_grad(self, min: f32, max: f32) -> Self {
        let op = PassThrough(
            self.ty(),
            Box::new(move |x| {
                let size = x.ty().size();
                let min = x.builder().scalar(min, size);
                let max = x.builder().scalar(max, size);
                x.max(min)?.min(max)
            }),
        );
        Self { node: self.builder.add_op([self], op)[0], ..self }
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
        let inner = self.nt.shape.size();
        let divisor = indices.shape().size();

        let op = Select { dtype, batch, inner, divisor };
        let rows = inner / divisor;
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
