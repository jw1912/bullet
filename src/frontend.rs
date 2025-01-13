use std::{
    collections::HashMap,
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
    sync::{Mutex, MutexGuard},
};

use crate::{
    autograd::{Graph, GraphBuilder, Node, Operation},
    operations, Activation, ConvolutionDescription, ExecutionContext, Shape,
};

#[derive(Clone, Copy, Debug)]
pub enum InitSettings {
    Normal(f32, f32),
    Uniform(f32, f32),
}

#[derive(Default)]
pub struct NetworkBuilder {
    graph_builder: Mutex<GraphBuilder>,
    init_data: Mutex<HashMap<String, InitSettings>>,
}

impl NetworkBuilder {
    fn builder(&self) -> MutexGuard<GraphBuilder> {
        self.graph_builder.try_lock().unwrap()
    }

    fn init(&self) -> MutexGuard<HashMap<String, InitSettings>> {
        self.init_data.try_lock().unwrap()
    }

    pub fn new_input<'a>(&'a self, id: &str, shape: Shape) -> NetworkNode<'a> {
        let node = self.builder().create_input(id, shape);
        NetworkNode { node, builder: self }
    }

    pub fn new_weights<'a>(&'a self, id: &str, shape: Shape) -> NetworkNode<'a> {
        let node = self.builder().create_weights(id, shape);
        NetworkNode { node, builder: self }
    }

    pub fn set_init_data(&self, id: String, init_data: InitSettings) {
        self.init().insert(id, init_data);
    }

    pub fn new_affine(&self, id: &str, input_size: usize, output_size: usize) -> Affine {
        let wid = format!("{}w", id);
        let weights = self.builder().create_weights(&wid, Shape::new(output_size, input_size));
        let bias = self.builder().create_weights(&format!("{}b", id), Shape::new(output_size, 1));

        self.set_init_data(wid, InitSettings::Uniform(0.0, 1.0 / (input_size as f32).sqrt()));

        Affine { weights, bias }
    }

    pub fn apply<'a>(&'a self, operation: impl Operation, inputs: &[Node]) -> NetworkNode<'a> {
        let node = self.builder().create_result_of_operation(operation, inputs);
        NetworkNode { node, builder: self }
    }

    pub fn build(self, execution_context: ExecutionContext) -> Graph {
        let builder = self.graph_builder.into_inner().unwrap();
        let mut graph = builder.build(execution_context);

        for (id, init_data) in self.init_data.lock().unwrap().iter() {
            let (mean, stdev, use_gaussian) = match *init_data {
                InitSettings::Normal(mean, stdev) => (mean, stdev, true),
                InitSettings::Uniform(mean, stdev) => (mean, stdev, false),
            };

            graph.get_weights_mut(id).seed_random(mean, stdev, use_gaussian);
        }

        graph
    }
}

#[derive(Clone, Copy)]
pub struct NetworkNode<'a> {
    node: Node,
    builder: &'a NetworkBuilder,
}

impl Add<Self> for NetworkNode<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.linear_comb(1.0, rhs, 1.0)
    }
}

impl AddAssign<Self> for NetworkNode<'_> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub<Self> for NetworkNode<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.linear_comb(1.0, rhs, -1.0)
    }
}

impl SubAssign<Self> for NetworkNode<'_> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<Self> for NetworkNode<'_> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.builder.apply(operations::Linear, &[self.node, rhs.node])
    }
}

impl NetworkNode<'_> {
    pub fn node(self) -> Node {
        self.node
    }

    pub fn activate(self, activation: Activation) -> Self {
        self.builder.apply(activation, &[self.node])
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
        self.builder.apply(desc, &[filters.node, self.node])
    }
}

#[derive(Clone, Copy)]
pub struct Affine {
    weights: Node,
    bias: Node,
}

impl Affine {
    pub fn forward(self, input: NetworkNode<'_>) -> NetworkNode<'_> {
        input.builder.apply(operations::Affine(operations::Linear), &[self.weights, input.node, self.bias])
    }

    pub fn forward_sparse_dual_with_activation<'a>(
        self,
        stm: NetworkNode<'a>,
        ntm: NetworkNode<'a>,
        activation: Activation,
    ) -> NetworkNode<'a> {
        stm.builder.apply(
            operations::SparseAffineDualWithActivation(activation),
            &[self.weights, stm.node, ntm.node, self.bias],
        )
    }
}
