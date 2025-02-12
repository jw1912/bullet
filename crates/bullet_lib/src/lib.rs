mod frontend;

/// Contains the `NetworkTrainer` trait and associated structs for its use
/// as well as the `default` impl of the trait for training value networks
pub mod trainer;

// TODO: Remove these re-exports as they are exported in the `nn` module
pub use bullet_core::{graph::operation::Activation, shape::Shape};
pub use bullet_hip_backend::ExecutionContext;

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
    pub use super::frontend::{Affine, InitSettings, NetworkBuilder, NetworkBuilderNode};

    pub use bullet_core::{
        graph::{builder::Node, operation::Activation},
        shape::Shape,
    };
    pub use bullet_hip_backend::ExecutionContext;
    pub type Graph = bullet_core::graph::Graph<ExecutionContext>;

    pub mod optimiser {
        use bullet_core::optimiser;
        use bullet_hip_backend::ExecutionContext;

        pub use bullet_core::optimiser::Optimiser;

        pub type AdamWOptimiser = optimiser::AdamW<ExecutionContext>;
        pub use optimiser::AdamWParams;

        pub trait OptimiserType: Default {
            type Optimiser: Optimiser<ExecutionContext>;
        }

        #[derive(Default)]
        pub struct AdamW;
        impl OptimiserType for AdamW {
            type Optimiser = AdamWOptimiser;
        }
    }
}
