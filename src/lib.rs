mod backend;
pub mod inputs;
pub mod loader;
pub mod operations;
pub mod optimiser;
pub mod outputs;
mod tensor;

pub use bulletformat as format;
pub use diffable::Node;
pub use tensor::Tensor;

pub type Graph = diffable::Graph<Tensor>;
pub type GraphBuilder = diffable::GraphBuilder<Tensor>;
