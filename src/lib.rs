mod autograd;
mod backend;
/// Contains the `DataLoader` trait:
/// - Determines how input files are read to produce the specified `BulletFormat` data type,
///     in order to support e.g. reading from binpacked data
/// - The `DirectSequentialDataLoader` is included to read all `BulletFormat` types directly
///     from input files
pub mod loader;
/// Contains functions that apply publically-exposed operations to nodes in the network graph.
pub mod operations;
/// Contains the `Optimiser` trait, for implementing custom optimisers, as well as all premade
/// optimisers that are commonly used (e.g. `AdamW`).
pub mod optimiser;
mod rng;
mod tensor;
mod trainer;

pub use autograd::{Graph, GraphBuilder, Node};
pub use backend::{ConvolutionDescription, ExecutionContext};
pub use bulletformat as format;
pub use tensor::{Activation, Shape};
pub use trainer::{
    default::{gamerunner, inputs, outputs, testing, Loss, QuantTarget, Trainer, TrainerBuilder},
    logger,
    schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
    settings::LocalSettings,
    DataPreparer, NetworkTrainer,
};
