use crate::graph::ir::operation::{sparse::SparseAffineActivate, unary::DiffableFromOutput};

use super::{Activation, GraphBuilderNode, InitSettings};

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

    pub fn forward_sparse_with_values(
        self,
        stm: GraphBuilderNode<'a>,
        vals: GraphBuilderNode<'a>,
    ) -> GraphBuilderNode<'a> {
        stm.builder.apply(SparseAffineActivate {
            weights: self.weights.node,
            indices: stm.node,
            values: Some(vals.node),
            biases: Some(self.bias.node),
            activation: DiffableFromOutput::Identity,
        })
    }
}
