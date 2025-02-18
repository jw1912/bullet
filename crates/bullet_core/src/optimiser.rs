mod adamw;
pub mod utils;

pub use adamw::{AdamW, AdamWParams};

use crate::{device::Device, graph::Graph, tensor::DenseMatrix};

pub trait OptimiserState<D: Device> {
    type Params: Clone + std::fmt::Debug + Default;

    fn new(graph: &Graph<D>, params: Self::Params) -> Self;

    fn update_single_weight(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        id: &str,
        gradient_factor: f32,
        learning_rate: f32,
    );

    fn reset(&mut self);

    fn load_from_checkpoint(&mut self, path: &str, old_format: bool);

    fn write_to_checkpoint(&self, path: &str);

    fn set_params_for_weight(&mut self, id: &str, params: Self::Params);
}

pub struct Optimiser<D: Device, S: OptimiserState<D>> {
    pub graph: Graph<D>,
    pub state: S,
}

impl<D: Device, S: OptimiserState<D>> Optimiser<D, S> {
    pub fn new(graph: Graph<D>, params: S::Params) -> Self {
        let state = S::new(&graph, params);
        Self { graph, state }
    }

    pub fn update(&mut self, gradient_factor: f32, learning_rate: f32) {
        for id in &self.graph.weight_ids() {
            let weights = self.graph.get_weights_mut(id);

            if let Some(grads) = weights.gradients.as_mut() {
                self.state.update_single_weight(weights.values.dense_mut(), grads, id, gradient_factor, learning_rate);
            }
        }
    }

    pub fn reset_state(&mut self) {
        self.state.reset();
    }

    pub fn set_params_for_weight(&mut self, id: &str, params: S::Params) {
        self.state.set_params_for_weight(id, params);
    }

    pub fn set_params(&mut self, params: S::Params) {
        for id in self.graph.weight_ids() {
            self.state.set_params_for_weight(&id, params.clone());
        }
    }

    pub fn write_to_checkpoint(&self, path: &str) {
        utils::write_graph_weights_to_file(&self.graph, &format!("{path}/weights.bin"));
        self.state.write_to_checkpoint(path);
    }

    pub fn load_from_checkpoint(&mut self, path: &str) {
        self.load_from_checkpoint_(path, false);
    }

    pub fn load_weights_from_file(&mut self, path: &str) {
        self.load_weights_from_file_(path, false);
    }

    pub fn load_from_old_format_checkpoint(&mut self, path: &str) {
        self.load_from_checkpoint_(path, true);
    }

    pub fn load_old_format_weights_from_file(&mut self, path: &str) {
        self.load_weights_from_file_(path, true);
    }

    fn load_weights_from_file_(&mut self, path: &str, old_format: bool) {
        utils::load_graph_weights_from_file(&mut self.graph, path, old_format);
    }

    fn load_from_checkpoint_(&mut self, path: &str, old_format: bool) {
        self.load_weights_from_file_(&format!("{path}/weights.bin"), old_format);
        self.state.load_from_checkpoint(path, old_format);
    }
}
