mod adamw;
pub mod utils;

pub use adamw::{AdamW, AdamWParams};

use crate::{device::Device, graph::Graph};

pub trait Optimiser<D: Device> {
    type Params: Clone + std::fmt::Debug + Default;

    fn new(graph: Graph<D>, params: Self::Params) -> Self;

    fn update(&mut self, gradient_factor: f32, learning_rate: f32);

    fn reset_state(&mut self);

    fn graph(&self) -> &Graph<D>;

    fn graph_mut(&mut self) -> &mut Graph<D>;

    fn load_from_checkpoint(&mut self, path: &str);

    fn write_to_checkpoint(&self, path: &str);

    fn set_params_for_weight(&mut self, id: &str, params: Self::Params);

    fn set_params(&mut self, params: Self::Params) {
        for id in self.graph().weight_ids() {
            self.set_params_for_weight(&id, params.clone());
        }
    }
}
