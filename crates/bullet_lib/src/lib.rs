/// Default implementation of `NetworkTrainer`,
/// for training value networks.
pub mod default;

pub mod policy;

/// Contains the `NetworkTrainer` trait and
/// associated structs for its use
pub mod trainer;

// TODO: Remove these re-exports as they are exported in the `nn` module
pub use bullet_core::graph::builder::{Activation, Shape};
pub use nn::ExecutionContext;

// TODO: Remove these re-exports as they are exported in the `trainer` module
pub use trainer::{
    schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
    settings::LocalSettings,
};

/// Re-export of `bullet_core`
pub use bullet_core as core;

/// Contains the Graph API, by which neural networks are created with
/// `NetworkBuilder`, and then compiled into an executable `Graph`
pub mod nn;
