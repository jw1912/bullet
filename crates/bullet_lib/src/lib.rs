mod frontend;
mod rng;

/// Contains the `NetworkTrainer` trait and associated structs for its use
/// as well as the `default` impl of the trait for training value networks
pub mod trainer;

/// Contains the `Optimiser` trait, for implementing custom optimisers, as well as all premade
/// optimisers that are commonly used (e.g. `AdamW`)
pub mod optimiser;

// TODO: Remove these re-exports as they are exported in the `nn` module
pub use bullet_backend::{Activation, ConvolutionDescription, ExecutionContext};
pub use bullet_core::shape::Shape;

// TODO: Remove these re-exports as they are exported in the `trainer` module
pub use trainer::{
    default, logger, save,
    schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
    settings::LocalSettings,
    DataPreparer, NetworkTrainer,
};

/// Contains the Graph API, by which neural networks are created with
/// `NetworkBuilder`, and then compiled into an executable `Graph`
pub mod nn {
    pub use super::{
        frontend::{Affine, InitSettings, NetworkBuilder, NetworkBuilderNode},
        optimiser,
    };

    pub use bullet_backend::{Activation, ConvolutionDescription, ExecutionContext};
    pub use bullet_core::{graph::Node, shape::Shape};
    pub type Graph = bullet_core::graph::Graph<ExecutionContext>;
}
