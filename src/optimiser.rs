mod adamw;
pub mod utils;

pub use adamw::{AdamW, AdamWParams};

use crate::Graph;

pub trait Optimiser {
    type Params: Clone + std::fmt::Debug;

    fn new(graph: Graph, params: Self::Params) -> Self;

    fn update(&mut self, gradient_factor: f32, learning_rate: f32);

    fn graph(&self) -> &Graph;

    fn graph_mut(&mut self) -> &mut Graph;

    fn load_from_checkpoint(&mut self, path: &str);

    fn write_to_checkpoint(&self, path: &str);
}
