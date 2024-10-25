mod backend;
/// Contains the `InputType` trait for implementing custom input types,
/// as well as several premade input formats that are commonly used.
pub mod inputs;
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
/// Contains the `OutputBuckets` trait for implementing custom output bucket types,
/// as well as several premade output buckets that are commonly used.
pub mod outputs;
mod rng;
mod tensor;
mod trainer;

pub use backend::ExecutionContext;
pub use bulletformat as format;
pub use diffable::Node;
pub use tensor::{Activation, Shape};
pub use trainer::{
    cutechess, logger,
    schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
    settings::LocalSettings,
    testing, DataPreparer, Loss, NetworkTrainer, QuantTarget, Trainer, TrainerBuilder,
};

pub type Graph = diffable::Graph<tensor::Tensor>;
pub type GraphBuilder = diffable::GraphBuilder<tensor::Tensor>;
