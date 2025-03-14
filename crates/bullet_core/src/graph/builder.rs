use std::{
    collections::HashMap,
    ops::{Add, Sub},
    sync::{Mutex, MutexGuard},
};

use crate::backend::device::{base::Activation, blas::Shape, Device};

use super::{
    ir::{args::GraphIRCompileArgs, node::AnnotatedNode, op::GraphIROp, GraphIR},
    Graph, Node,
};

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
        match self.builder().add_op(operation, true) {
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
        let init = InitSettings::Normal { mean: 0.0, stdev: 1.0 / (input_size as f32 * bias_cols as f32).sqrt() };
        let weights = self.new_weights(&wid, Shape::new(output_size, input_size), init);
        let bias = self.new_weights(&format!("{}b", id), Shape::new(output_size, bias_cols), InitSettings::Zeroed);

        Affine { weights: weights.node, bias: bias.node }
    }

    pub fn set_compile_args(&mut self, args: GraphIRCompileArgs) {
        self.args = args;
    }

    pub fn build<D: Device>(self, device: D) -> Graph<D> {
        let mut builder = self.graph_builder.into_inner().unwrap();
        builder.add_op(GraphIROp::ReduceAcrossBatch(builder.root()), true).unwrap();
        builder.optimise();
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

impl GraphBuilderNode<'_> {
    pub fn node(self) -> Node {
        self.node.into()
    }

    pub fn reshape(mut self, shape: Shape) -> Self {
        self.node = self.node.reshape(shape).unwrap();
        self
    }

    pub fn activate(self, activation: Activation) -> Self {
        self.builder.apply(GraphIROp::Activate(self.node, activation))
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
        if rhs.node.is_sparse() {
            self.builder.apply(GraphIROp::SparseAffine(self.node, rhs.node, None))
        } else {
            self.builder.apply(GraphIROp::Matmul(self.node, false, rhs.node, false))
        }
    }

    pub fn gemm(self, transa: bool, rhs: Self, transb: bool) -> Self {
        self.builder.apply(GraphIROp::Matmul(self.node, transa, rhs.node, transb))
    }

    pub fn mpe(self, targets: Self, power: f32) -> Self {
        self.builder.apply(GraphIROp::PowerError(self.node, targets.node, power))
    }

    pub fn mse(self, targets: Self) -> Self {
        self.mpe(targets, 2.0)
    }

    pub fn pairwise_mul(self) -> Self {
        self.builder.apply(GraphIROp::PairwiseMul(self.node, false))
    }

    pub fn pairwise_mul_post_affine_dual(self) -> Self {
        self.builder.apply(GraphIROp::PairwiseMul(self.node, true))
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
        let node = self.builder.builder().add_op(GraphIROp::ToDense(self.node), false).unwrap();
        Self { node, builder: self.builder }
    }
}

#[derive(Clone, Copy)]
pub struct Affine {
    pub weights: AnnotatedNode,
    pub bias: AnnotatedNode,
}

impl Affine {
    pub fn forward(self, input: GraphBuilderNode<'_>) -> GraphBuilderNode<'_> {
        if input.node.is_sparse() {
            input.builder.apply(GraphIROp::SparseAffine(self.weights, input.node, Some(self.bias)))
        } else {
            input.builder.apply(GraphIROp::Affine(self.weights, input.node, self.bias))
        }
    }

    pub fn forward_sparse_dual_with_activation<'a>(
        self,
        stm: GraphBuilderNode<'a>,
        ntm: GraphBuilderNode<'a>,
        activation: Activation,
    ) -> GraphBuilderNode<'a> {
        stm.builder.apply(GraphIROp::SparseAffineDualActivate(self.weights, stm.node, ntm.node, self.bias, activation))
    }

    pub fn forward_sparse_dual_with_activation_and_bias_buckets<'a>(
        self,
        stm: GraphBuilderNode<'a>,
        ntm: GraphBuilderNode<'a>,
        buckets: GraphBuilderNode<'a>,
        activation: Activation,
    ) -> GraphBuilderNode<'a> {
        let biases = stm.builder.apply(GraphIROp::SparseAffine(self.bias, buckets.node, None));
        stm.builder.apply(GraphIROp::SparseAffineDualActivate(
            self.weights,
            stm.node,
            ntm.node,
            biases.node,
            activation,
        ))
    }
}
