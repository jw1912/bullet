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
        use bullet_core::optimiser::{self, clip, decay, radam, utils::Placement, OptimiserState};
        use bullet_hip_backend::ExecutionContext;

        pub type AdamWOptimiser = optimiser::AdamW<ExecutionContext>;
        pub type RAdamOptimiser = clip::WeightClipping<decay::WeightDecay<radam::RAdam<ExecutionContext>>>;
        pub use optimiser::AdamWParams;

        pub trait OptimiserType: Default {
            type Optimiser: OptimiserState<ExecutionContext>;
        }

        #[derive(Default)]
        pub struct AdamW;
        impl OptimiserType for AdamW {
            type Optimiser = AdamWOptimiser;
        }

        #[derive(Default)]
        pub struct RAdam;
        impl OptimiserType for RAdam {
            type Optimiser = RAdamOptimiser;
        }

        #[derive(Clone, Copy, Debug)]
        pub struct RAdamParams {
            pub decay: f32,
            pub beta1: f32,
            pub beta2: f32,
            pub min_weight: f32,
            pub max_weight: f32,
        }

        impl From<RAdamParams> for clip::WeightClippingParams<decay::WeightDecayParams<radam::RAdamParams>> {
            fn from(value: RAdamParams) -> Self {
                clip::WeightClippingParams {
                    inner: decay::WeightDecayParams {
                        inner: radam::RAdamParams { beta1: value.beta1, beta2: value.beta2, n_sma_threshold: 5.0 },
                        placement: Placement::Before,
                        decay: value.decay,
                    },
                    placement: Placement::After,
                    min: value.min_weight,
                    max: value.max_weight,
                }
            }
        }
    }
}
