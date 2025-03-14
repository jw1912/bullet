/// Contains the `NetworkTrainer` trait and associated structs for its use
/// as well as the `default` impl of the trait for training value networks
pub mod trainer;

// TODO: Remove these re-exports as they are exported in the `nn` module
pub use bullet_core::backend::device::{base::Activation, blas::Shape};
pub use nn::ExecutionContext;

// TODO: Remove these re-exports as they are exported in the `trainer` module
pub use trainer::{
    default, logger, save,
    schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
    settings::LocalSettings,
    DataPreparer, NetworkTrainer,
};

/// Re-export of `bullet_core`
pub use bullet_core as core;

/// Contains the Graph API, by which neural networks are created with
/// `NetworkBuilder`, and then compiled into an executable `Graph`
pub mod nn;
