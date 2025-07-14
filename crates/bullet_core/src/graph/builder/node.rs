use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::graph::{
    builder::GraphBuilder,
    ir::{
        node::AnnotatedNode,
        operation::{
            affine::Matmul,
            binary::{Concat, LinearCombination, Select, SoftmaxCrossEntropy},
            sparse::SparseAffineActivate,
            unary::{Copy, DiffableFromOutput, PairwiseMul, Reduce, ReduceAcrossBatch, Slice, ToDense, Unary, UnaryOp},
        },
        shape::Shape,
        BackendMarker,
    },
    Node,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Identity,
    ReLU,
    CReLU,
    SCReLU,
    SqrReLU,
    Sigmoid,
    Square,
}

#[derive(Clone, Copy)]
pub struct GraphBuilderNode<'a, B: BackendMarker> {
    pub(super) node: AnnotatedNode,
    pub(super) builder: &'a GraphBuilder<B>,
}

impl<B: BackendMarker> Add<Self> for GraphBuilderNode<'_, B> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.linear_comb(1.0, rhs, 1.0)
    }
}

impl<B: BackendMarker> Sub<Self> for GraphBuilderNode<'_, B> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.linear_comb(1.0, rhs, -1.0)
    }
}

impl<'a, B: BackendMarker> Add<GraphBuilderNode<'a, B>> for f32 {
    type Output = GraphBuilderNode<'a, B>;

    fn add(self, rhs: GraphBuilderNode<'a, B>) -> Self::Output {
        rhs.builder.apply(Unary { input: rhs.node, op: UnaryOp::Add(self) })
    }
}

impl<B: BackendMarker> Add<f32> for GraphBuilderNode<'_, B> {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        rhs + self
    }
}

impl<B: BackendMarker> Neg for GraphBuilderNode<'_, B> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -1.0 * self
    }
}

impl<'a, B: BackendMarker> Sub<GraphBuilderNode<'a, B>> for f32 {
    type Output = GraphBuilderNode<'a, B>;

    fn sub(self, rhs: GraphBuilderNode<'a, B>) -> Self::Output {
        self + (-1.0 * rhs)
    }
}

impl<B: BackendMarker> Sub<f32> for GraphBuilderNode<'_, B> {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        self + (-rhs)
    }
}

impl<'a, B: BackendMarker> Mul<GraphBuilderNode<'a, B>> for f32 {
    type Output = GraphBuilderNode<'a, B>;

    fn mul(self, rhs: GraphBuilderNode<'a, B>) -> Self::Output {
        rhs.builder.apply(Unary { input: rhs.node, op: UnaryOp::Mul(self) })
    }
}

impl<'a, B: BackendMarker> Mul<GraphBuilderNode<'a, B>> for GraphBuilderNode<'a, B> {
    type Output = GraphBuilderNode<'a, B>;

    fn mul(self, rhs: GraphBuilderNode<'a, B>) -> Self::Output {
        self.concat(rhs).pairwise_mul()
    }
}

impl<B: BackendMarker> Mul<f32> for GraphBuilderNode<'_, B> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        rhs * self
    }
}

impl<B: BackendMarker> Div<f32> for GraphBuilderNode<'_, B> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        (1.0 / rhs) * self
    }
}

impl<B: BackendMarker> GraphBuilderNode<'_, B> {
    pub fn annotated_node(&self) -> AnnotatedNode {
        self.node
    }

    pub fn node(self) -> Node {
        self.node.into()
    }

    pub fn reshape(mut self, shape: Shape) -> Self {
        self.node = self.node.reshape(shape).unwrap();
        self
    }

    pub fn relu(self) -> Self {
        self.diffable_from_output(DiffableFromOutput::ReLU)
    }

    pub fn crelu(self) -> Self {
        self.diffable_from_output(DiffableFromOutput::CReLU)
    }

    pub fn screlu(self) -> Self {
        self.diffable_from_output(DiffableFromOutput::SCReLU)
    }

    pub fn sqrrelu(self) -> Self {
        self.diffable_from_output(DiffableFromOutput::SqrReLU)
    }

    pub fn sigmoid(self) -> Self {
        self.diffable_from_output(DiffableFromOutput::Sigmoid)
    }

    fn diffable_from_output(self, act: DiffableFromOutput) -> Self {
        self.builder.apply(Unary { input: self.node, op: UnaryOp::DiffableFromOutput(act) })
    }

    pub fn activate(self, act: Activation) -> Self {
        match act {
            Activation::Identity => self.diffable_from_output(DiffableFromOutput::Identity),
            Activation::ReLU => self.relu(),
            Activation::CReLU => self.crelu(),
            Activation::SCReLU => self.screlu(),
            Activation::Sigmoid => self.sigmoid(),
            Activation::SqrReLU => self.sqrrelu(),
            Activation::Square => self.abs_pow(2.0),
        }
    }

    pub fn select(self, buckets: Self) -> Self {
        self.builder.apply(Select { input: self.node, buckets: buckets.node })
    }

    pub fn concat(self, rhs: Self) -> Self {
        self.builder.apply(Concat { a: self.node, b: rhs.node })
    }

    pub fn copy_stop_grad(self) -> Self {
        self.builder.apply(Copy { input: self.node, stop_grad: true })
    }

    pub fn copy(self) -> Self {
        self.builder.apply(Copy { input: self.node, stop_grad: false })
    }

    pub fn linear_comb(self, alpha: f32, rhs: Self, beta: f32) -> Self {
        self.builder.apply(LinearCombination { a: self.node, b: rhs.node, alpha, beta })
    }

    pub fn reduce_sum_across_batch(self) -> Self {
        self.builder.apply(ReduceAcrossBatch { input: self.node, reduction: Reduce::Sum })
    }

    pub fn reduce_avg_across_batch(self) -> Self {
        self.builder.apply(ReduceAcrossBatch { input: self.node, reduction: Reduce::Avg })
    }

    pub fn matmul(self, rhs: Self) -> Self {
        if self.builder.ir().get(rhs.node.idx).unwrap().info.sparse.is_some() {
            self.builder.apply(SparseAffineActivate {
                weights: self.node,
                indices: rhs.node,
                values: None,
                biases: None,
                activation: DiffableFromOutput::Identity,
            })
        } else {
            self.builder.apply(Matmul { a: self.node, transa: false, b: rhs.node, transb: false })
        }
    }

    pub fn gemm(self, transa: bool, rhs: Self, transb: bool) -> Self {
        self.builder.apply(Matmul { a: self.node, transa, b: rhs.node, transb })
    }

    pub fn power_error(self, targets: Self, power: f32) -> Self {
        self.builder.apply(Unary { input: (self - targets).node, op: UnaryOp::AbsPow(power) })
    }

    pub fn squared_error(self, targets: Self) -> Self {
        self.power_error(targets, 2.0)
    }

    #[deprecated]
    pub fn mpe(self, targets: Self, power: f32) -> Self {
        self.power_error(targets, power)
    }

    pub fn repeat(self, n: usize) -> Self {
        let shape = self.node.shape;
        let ones = self.builder.new_constant(Shape::new(1, n), &vec![1.0; n]);
        let resh = self.reshape(Shape::new(shape.size(), 1));
        let reps = resh.matmul(ones);
        reps.reshape(Shape::new(shape.rows(), shape.cols() * n))
    }

    #[deprecated]
    pub fn mse(self, targets: Self) -> Self {
        self.squared_error(targets)
    }

    pub fn pairwise_mul(self) -> Self {
        self.builder.apply(PairwiseMul { input: self.node, post_concat: false })
    }

    pub fn pairwise_mul_post_affine_dual(self) -> Self {
        self.builder.apply(PairwiseMul { input: self.node, post_concat: true })
    }

    pub fn abs_pow(self, power: f32) -> Self {
        self.builder.apply(Unary { input: self.node, op: UnaryOp::AbsPow(power) })
    }

    pub fn softmax_crossentropy_loss(self, targets: Self) -> Self {
        self.builder.apply(SoftmaxCrossEntropy { logits: self.annotated_node(), targets: targets.annotated_node() })
    }

    pub fn slice_rows(self, start: usize, end: usize) -> Self {
        self.builder.apply(Slice { input: self.node, start, end })
    }

    pub fn to_dense(self) -> Self {
        let node = self.builder.ir().add_op(ToDense(self.node)).unwrap();
        Self { node, builder: self.builder }
    }
}
