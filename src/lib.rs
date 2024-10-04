mod backend;
pub mod inputs;
pub mod loader;
pub mod operations;
pub mod optimiser;
pub mod outputs;
pub mod rng;
mod tensor;
mod trainer;

pub use backend::ExecutionContext;
pub use bulletformat as format;
pub use diffable::Node;
pub use tensor::{Activation, Shape, Tensor};
pub use trainer::{
    cutechess,
    logger,
    schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
    settings::LocalSettings,
    testing,
    NetworkTrainer, Trainer, TrainerBuilder, Loss,
};

pub type Graph = diffable::Graph<Tensor>;
pub type GraphBuilder = diffable::GraphBuilder<Tensor>;
