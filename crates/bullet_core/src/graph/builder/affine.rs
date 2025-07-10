use crate::graph::ir::{
    operation::{sparse::SparseAffineActivate, unary::DiffableFromOutput},
    BackendMarker,
};

use super::{Activation, GraphBuilderNode, InitSettings};

#[derive(Clone, Copy)]
pub struct Affine<'a, B: BackendMarker> {
    pub weights: GraphBuilderNode<'a, B>,
    pub bias: GraphBuilderNode<'a, B>,
}

impl<'a, B: BackendMarker> Affine<'a, B> {
    pub fn forward(self, input: GraphBuilderNode<'a, B>) -> GraphBuilderNode<'a, B> {
        self.weights.matmul(input) + self.bias
    }

    pub fn init_with_effective_input_size(&self, size: usize) {
        let builder = self.weights.builder.ir();
        let w = builder.get(self.weights.node.idx).unwrap();
        let id = w.id.clone().unwrap();
        *self.weights.builder.init().get_mut(&id).unwrap() =
            InitSettings::Normal { mean: 0.0, stdev: (2.0 / size as f32).sqrt() };
    }

    pub fn forward_sparse_dual_with_activation(
        self,
        stm: GraphBuilderNode<'a, B>,
        ntm: GraphBuilderNode<'a, B>,
        activation: Activation,
    ) -> GraphBuilderNode<'a, B> {
        self.forward(stm).concat(self.forward(ntm)).activate(activation)
    }

    pub fn forward_sparse_with_values(
        self,
        stm: GraphBuilderNode<'a, B>,
        vals: GraphBuilderNode<'a, B>,
    ) -> GraphBuilderNode<'a, B> {
        stm.builder.apply(SparseAffineActivate {
            weights: self.weights.node,
            indices: stm.node,
            values: Some(vals.node),
            biases: Some(self.bias.node),
            activation: DiffableFromOutput::Identity,
        })
    }
}
