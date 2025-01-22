mod adamw;
pub mod utils;

pub use adamw::{AdamW, AdamWOptimiser, AdamWParams};

use crate::nn::Graph;

pub trait Optimiser {
    type Params: Clone + std::fmt::Debug + Default;

    fn new(graph: Graph, params: Self::Params) -> Self;

    fn update(&mut self, gradient_factor: f32, learning_rate: f32);

    fn graph(&self) -> &Graph;

    fn graph_mut(&mut self) -> &mut Graph;

    fn load_from_checkpoint(&mut self, path: &str);

    fn write_to_checkpoint(&self, path: &str);

    fn set_params_for_weight(&mut self, id: &str, params: Self::Params);

    fn set_params(&mut self, params: Self::Params) {
        for id in self.graph().weight_ids() {
            self.set_params_for_weight(&id, params.clone());
        }
    }
}

/// This is for use in `TrainerBuilder`
pub trait OptimiserType: Default {
    type Optimiser: Optimiser;
}
