mod backend;
pub mod inputs;
mod loader;
pub mod operations;
pub mod optimiser;
mod outputs;
mod tensor;
mod trainer;

pub use bulletformat as format;
pub use diffable::Node;
pub use loader::{DataLoader, DirectSequentialDataLoader};
pub use tensor::Tensor;
pub use trainer::{
    schedule::{lr, wdl, TrainingSchedule},
    settings::LocalSettings,
    NetworkTrainer,
};

pub type Graph = diffable::Graph<Tensor>;
pub type GraphBuilder = diffable::GraphBuilder<Tensor>;
