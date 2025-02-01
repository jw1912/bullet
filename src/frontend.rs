use std::{
    collections::HashMap,
    ops::{Add, Sub},
    sync::{Mutex, MutexGuard},
};

use bullet_backend::operations;
use bullet_core::graph::{Graph, GraphBuilder, Node, Operation};

use crate::{Activation, ConvolutionDescription, ExecutionContext, Shape};

#[derive(Clone, Copy, Debug)]
pub enum InitSettings {
    Zeroed,
    Normal { mean: f32, stdev: f32 },
    Uniform { mean: f32, stdev: f32 },
}

#[derive(Default)]
pub struct NetworkBuilder {
    graph_builder: Mutex<GraphBuilder<ExecutionContext>>,
    init_data: Mutex<HashMap<String, InitSettings>>,
}

impl NetworkBuilder {
    fn builder(&self) -> MutexGuard<GraphBuilder<ExecutionContext>> {
        self.graph_builder.try_lock().unwrap()
    }

    fn init(&self) -> MutexGuard<HashMap<String, InitSettings>> {
        self.init_data.try_lock().unwrap()
    }

    pub fn new_input<'a>(&'a self, id: &str, shape: Shape) -> NetworkBuilderNode<'a> {
        let node = self.builder().create_input(id, shape);
        NetworkBuilderNode { node, builder: self }
    }

    pub fn new_weights<'a>(&'a self, id: &str, shape: Shape, init: InitSettings) -> NetworkBuilderNode<'a> {
        let node = self.builder().create_weights(id, shape);
        self.init().insert(id.to_string(), init);
        NetworkBuilderNode { node, builder: self }
    }

    pub fn new_affine(&self, id: &str, input_size: usize, output_size: usize) -> Affine {
        let wid = format!("{}w", id);
        let init = InitSettings::Uniform { mean: 0.0, stdev: 1.0 / (input_size as f32).sqrt() };
        let weights = self.new_weights(&wid, Shape::new(output_size, input_size), init);
        let bias = self.new_weights(&format!("{}b", id), Shape::new(output_size, 1), InitSettings::Zeroed);

        Affine { weights: weights.node, bias: bias.node }
    }

    pub fn apply<'a>(&'a self, operation: impl Operation<ExecutionContext>, inputs: &[Node]) -> NetworkBuilderNode<'a> {
        let node = self.builder().create_result_of_operation(operation, inputs);
        NetworkBuilderNode { node, builder: self }
    }

    pub fn build(self, execution_context: ExecutionContext) -> Graph<ExecutionContext> {
        let mut builder = self.graph_builder.into_inner().unwrap();
        builder.create_result_of_operation(operations::ReduceAcrossBatch, &[builder.root()]);
        let mut graph = builder.build(execution_context);

        for (id, init_data) in self.init_data.lock().unwrap().iter() {
            match *init_data {
                InitSettings::Zeroed => {}
                InitSettings::Normal { mean, stdev } => graph.get_weights_mut(id).seed_random(mean, stdev, true),
                InitSettings::Uniform { mean, stdev } => graph.get_weights_mut(id).seed_random(mean, stdev, false),
            };
        }

        graph
    }
}

#[derive(Clone, Copy)]
pub struct NetworkBuilderNode<'a> {
    node: Node,
    builder: &'a NetworkBuilder,
}

impl Add<Self> for NetworkBuilderNode<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.linear_comb(1.0, rhs, 1.0)
    }
}

impl Sub<Self> for NetworkBuilderNode<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.linear_comb(1.0, rhs, -1.0)
    }
}

impl NetworkBuilderNode<'_> {
    pub fn node(self) -> Node {
        self.node
    }

    pub fn activate(self, activation: Activation) -> Self {
        self.builder.apply(operations::Activate(activation), &[self.node])
    }

    pub fn select(self, buckets: Self) -> Self {
        self.builder.apply(operations::Select, &[self.node, buckets.node])
    }

    pub fn concat(self, rhs: Self) -> Self {
        self.builder.apply(operations::Concat, &[self.node, rhs.node])
    }

    pub fn linear_comb(self, alpha: f32, rhs: Self, beta: f32) -> Self {
        self.builder.apply(operations::LinearCombination(alpha, beta), &[self.node, rhs.node])
    }

    pub fn matmul(self, rhs: Self) -> Self {
        self.builder.apply(operations::Linear, &[self.node, rhs.node])
    }

    pub fn mpe(self, targets: Self, power: f32) -> Self {
        self.builder.apply(operations::AbsPowerError(power), &[self.node, targets.node])
    }

    pub fn mse(self, targets: Self) -> Self {
        self.mpe(targets, 2.0)
    }

    pub fn pairwise_mul(self) -> Self {
        self.builder.apply(operations::PairwiseMul(false), &[self.node])
    }

    pub fn pairwise_mul_post_affine_dual(self) -> Self {
        self.builder.apply(operations::PairwiseMul(true), &[self.node])
    }

    pub fn mask(self, mask: Self) -> Self {
        self.builder.apply(operations::Mask, &[self.node, mask.node])
    }

    pub fn gather(self, indices: Self) -> Self {
        self.builder.apply(operations::Gather, &[self.node, indices.node])
    }

    pub fn submatrix_product(self, rhs: Self, size: usize) -> Self {
        self.builder.apply(operations::SubmatrixProduct(size), &[self.node, rhs.node])
    }

    pub fn softmax_crossentropy_loss(self, targets: Self) -> Self {
        self.builder.apply(operations::SoftmaxCrossEntropyLoss, &[self.node, targets.node])
    }

    pub fn masked_softmax_crossentropy_loss(self, targets: Self, mask: Self) -> Self {
        self.builder.apply(operations::SparseSoftmaxCrossEntropyLoss, &[mask.node, self.node, targets.node])
    }

    pub fn slice_rows(self, start: usize, end: usize) -> Self {
        self.builder.apply(operations::SliceRows(start, end), &[self.node])
    }

    pub fn convolution(self, filters: Self, desc: ConvolutionDescription) -> Self {
        self.builder.apply(operations::Convolution(desc), &[filters.node, self.node])
    }
}

#[derive(Clone, Copy)]
pub struct Affine {
    weights: Node,
    bias: Node,
}

impl Affine {
    pub fn forward(self, input: NetworkBuilderNode<'_>) -> NetworkBuilderNode<'_> {
        input.builder.apply(operations::Affine(operations::Linear), &[self.weights, input.node, self.bias])
    }

    pub fn forward_sparse_dual_with_activation<'a>(
        self,
        stm: NetworkBuilderNode<'a>,
        ntm: NetworkBuilderNode<'a>,
        activation: Activation,
    ) -> NetworkBuilderNode<'a> {
        stm.builder.apply(operations::AffineDualActivate(activation), &[self.weights, stm.node, ntm.node, self.bias])
    }
}
