mod backend;
pub mod inputs;
pub mod loader;
pub mod operations;
pub mod optimiser;
pub mod outputs;
mod tensor;
mod trainer;

pub use backend::ExecutionContext;
pub use bulletformat as format;
pub use diffable::Node;
pub use tensor::{Shape, Tensor};
pub use trainer::{
    logger,
    schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
    settings::LocalSettings,
    NetworkTrainer, Trainer,
};

pub type Graph = diffable::Graph<Tensor>;
pub type GraphBuilder = diffable::GraphBuilder<Tensor>;
