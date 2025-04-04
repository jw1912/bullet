use std::{
    collections::HashMap,
    ops::{Add, Div, Mul, Sub},
    sync::{Mutex, MutexGuard},
};

use crate::backend::device::Device;

use super::{
    ir::{
        args::GraphIRCompileArgs,
        node::AnnotatedNode,
        op::{DiffableFromOutput, GraphIROp, UnaryOp},
        GraphIR,
    },
    Graph, Node,
};

pub use crate::graph::ir::shape::Shape;

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

#[derive(Clone, Copy, Debug)]
pub enum InitSettings {
    Zeroed,
    Normal { mean: f32, stdev: f32 },
    Uniform { mean: f32, stdev: f32 },
}

#[derive(Default)]
pub struct GraphBuilder {
    graph_builder: Mutex<GraphIR>,
    init_data: Mutex<HashMap<String, InitSettings>>,
    args: GraphIRCompileArgs,
}

impl GraphBuilder {
    fn builder(&self) -> MutexGuard<GraphIR> {
        self.graph_builder.try_lock().unwrap()
    }

    fn init(&self) -> MutexGuard<HashMap<String, InitSettings>> {
        self.init_data.try_lock().unwrap()
    }

    fn apply(&self, operation: GraphIROp) -> GraphBuilderNode {
        match self.builder().add_op(operation) {
            Ok(node) => GraphBuilderNode { node, builder: self },
            Err(e) => {
                println!("{e:#?}");
                panic!();
            }
        }
    }

    pub fn new_dense_input<'a>(&'a self, id: &str, shape: Shape) -> GraphBuilderNode<'a> {
        let node = self.builder().add_dense_input(id, shape).unwrap();
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_sparse_input<'a>(&'a self, id: &str, shape: Shape, nnz: usize) -> GraphBuilderNode<'a> {
        let node = self.builder().add_sparse_input(id, shape, nnz).unwrap();
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_weights<'a>(&'a self, id: &str, shape: Shape, init: InitSettings) -> GraphBuilderNode<'a> {
        let node = self.builder().add_weights(id, shape).unwrap();
        self.init().insert(id.to_string(), init);
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_affine(&self, id: &str, input_size: usize, output_size: usize) -> Affine {
        self.new_affine_custom(id, input_size, output_size, 1)
    }

    pub fn new_affine_custom(&self, id: &str, input_size: usize, output_size: usize, bias_cols: usize) -> Affine {
        let wid = format!("{}w", id);
        let init = InitSettings::Normal { mean: 0.0, stdev: (2.0 / (input_size as f32 * bias_cols as f32)).sqrt() };
        let weights = self.new_weights(&wid, Shape::new(output_size, input_size), init);
        let bias = self.new_weights(&format!("{}b", id), Shape::new(output_size, bias_cols), InitSettings::Zeroed);

        Affine { weights, bias }
    }

    pub fn set_compile_args(&mut self, args: GraphIRCompileArgs) {
        self.args = args;
    }

    pub fn build<D: Device>(self, device: D) -> Graph<D> {
        let mut builder = self.graph_builder.into_inner().unwrap();
        builder.add_op(GraphIROp::ReduceAcrossBatch(builder.root().unwrap())).unwrap();
        let mut graph = builder.compile(device, self.args).unwrap();

        for (id, init_data) in self.init_data.lock().unwrap().iter() {
            match *init_data {
                InitSettings::Zeroed => {}
                InitSettings::Normal { mean, stdev } => {
                    graph.get_weights_mut(id).seed_random(mean, stdev, true).unwrap()
                }
                InitSettings::Uniform { mean, stdev } => {
                    graph.get_weights_mut(id).seed_random(mean, stdev, false).unwrap()
                }
            };
        }

        graph
    }
}

#[derive(Clone, Copy)]
pub struct GraphBuilderNode<'a> {
    node: AnnotatedNode,
    builder: &'a GraphBuilder,
}

impl Add<Self> for GraphBuilderNode<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.linear_comb(1.0, rhs, 1.0)
    }
}

impl Sub<Self> for GraphBuilderNode<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.linear_comb(1.0, rhs, -1.0)
    }
}

impl<'a> Add<GraphBuilderNode<'a>> for f32 {
    type Output = GraphBuilderNode<'a>;

    fn add(self, rhs: GraphBuilderNode<'a>) -> Self::Output {
        rhs.builder.apply(GraphIROp::Unary(rhs.node, UnaryOp::Add(self)))
    }
}

impl Add<f32> for GraphBuilderNode<'_> {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        rhs + self
    }
}

impl<'a> Sub<GraphBuilderNode<'a>> for f32 {
    type Output = GraphBuilderNode<'a>;

    fn sub(self, rhs: GraphBuilderNode<'a>) -> Self::Output {
        self + (-1.0 * rhs)
    }
}

impl Sub<f32> for GraphBuilderNode<'_> {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        self + (-rhs)
    }
}

impl<'a> Mul<GraphBuilderNode<'a>> for f32 {
    type Output = GraphBuilderNode<'a>;

    fn mul(self, rhs: GraphBuilderNode<'a>) -> Self::Output {
        rhs.builder.apply(GraphIROp::Unary(rhs.node, UnaryOp::Mul(self)))
    }
}

impl Mul<f32> for GraphBuilderNode<'_> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        rhs * self
    }
}

impl Div<f32> for GraphBuilderNode<'_> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        (1.0 / rhs) * self
    }
}

impl GraphBuilderNode<'_> {
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
        self.builder.apply(GraphIROp::Unary(self.node, UnaryOp::DiffableFromOutput(act)))
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
        self.builder.apply(GraphIROp::Select(self.node, buckets.node))
    }

    pub fn concat(self, rhs: Self) -> Self {
        self.builder.apply(GraphIROp::Concat(self.node, rhs.node))
    }

    pub fn linear_comb(self, alpha: f32, rhs: Self, beta: f32) -> Self {
        self.builder.apply(GraphIROp::LinearCombination(alpha, self.node, beta, rhs.node))
    }

    pub fn matmul(self, rhs: Self) -> Self {
        if self.builder.builder().get(rhs.node.idx).unwrap().sparse.is_some() {
            self.builder.apply(GraphIROp::SparseAffineActivate(self.node, rhs.node, None, DiffableFromOutput::Identity))
        } else {
            self.builder.apply(GraphIROp::Matmul(self.node, false, rhs.node, false))
        }
    }

    pub fn gemm(self, transa: bool, rhs: Self, transb: bool) -> Self {
        self.builder.apply(GraphIROp::Matmul(self.node, transa, rhs.node, transb))
    }

    pub fn power_error(self, targets: Self, power: f32) -> Self {
        self.builder.apply(GraphIROp::Unary((self - targets).node, UnaryOp::AbsPow(power)))
    }

    pub fn squared_error(self, targets: Self) -> Self {
        self.power_error(targets, 2.0)
    }

    #[deprecated]
    pub fn mpe(self, targets: Self, power: f32) -> Self {
        self.power_error(targets, power)
    }

    #[deprecated]
    pub fn mse(self, targets: Self) -> Self {
        self.squared_error(targets)
    }

    pub fn pairwise_mul(self) -> Self {
        self.builder.apply(GraphIROp::PairwiseMul(self.node, false))
    }

    pub fn pairwise_mul_post_affine_dual(self) -> Self {
        self.builder.apply(GraphIROp::PairwiseMul(self.node, true))
    }

    pub fn abs_pow(self, power: f32) -> Self {
        self.builder.apply(GraphIROp::Unary(self.node, UnaryOp::AbsPow(power)))
    }

    pub fn mask(self, mask: Self) -> Self {
        self.builder.apply(GraphIROp::Mask(self.node, mask.node))
    }

    pub fn gather(self, indices: Self) -> Self {
        self.builder.apply(GraphIROp::Gather(self.node, indices.node))
    }

    pub fn softmax_crossentropy_loss(self, targets: Self) -> Self {
        self.builder.apply(GraphIROp::SoftmaxCrossEntropyLoss(self.node, targets.node))
    }

    pub fn masked_softmax_crossentropy_loss(self, targets: Self, mask: Self) -> Self {
        self.builder.apply(GraphIROp::MaskedSoftmaxCrossEntropyLoss(mask.node, self.node, targets.node))
    }

    pub fn slice_rows(self, start: usize, end: usize) -> Self {
        self.builder.apply(GraphIROp::Slice(self.node, start, end))
    }

    pub fn to_dense(self) -> Self {
        let node = self.builder.builder().add_op(GraphIROp::ToDense(self.node)).unwrap();
        Self { node, builder: self.builder }
    }
}

#[derive(Clone, Copy)]
pub struct Affine<'a> {
    pub weights: GraphBuilderNode<'a>,
    pub bias: GraphBuilderNode<'a>,
}

impl<'a> Affine<'a> {
    pub fn forward(self, input: GraphBuilderNode<'a>) -> GraphBuilderNode<'a> {
        self.weights.matmul(input) + self.bias
    }

    pub fn init_with_effective_input_size(&self, size: usize) {
        let builder = self.weights.builder.builder();
        let w = builder.get(self.weights.node.idx).unwrap();
        let id = w.id.clone().unwrap();
        *self.weights.builder.init().get_mut(&id).unwrap() =
            InitSettings::Normal { mean: 0.0, stdev: (2.0 / size as f32).sqrt() };
    }

    pub fn forward_sparse_dual_with_activation(
        self,
        stm: GraphBuilderNode<'a>,
        ntm: GraphBuilderNode<'a>,
        activation: Activation,
    ) -> GraphBuilderNode<'a> {
        self.forward(stm).concat(self.forward(ntm)).activate(activation)
    }
}
