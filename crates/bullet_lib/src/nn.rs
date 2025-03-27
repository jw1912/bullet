pub use bullet_core::graph::{
    builder::{
        Activation, Affine, GraphBuilder as NetworkBuilder, GraphBuilderNode as NetworkBuilderNode, InitSettings, Shape,
    },
    ir::args::GraphIRCompileArgs as GraphCompileArgs,
    Node,
};
pub type Graph = bullet_core::graph::Graph<ExecutionContext>;

#[cfg(feature = "cpu")]
pub use bullet_core::backend::cpu::{CpuError as DeviceError, CpuThread as ExecutionContext};

#[cfg(not(feature = "cpu"))]
pub use bullet_hip_backend::{DeviceError, ExecutionContext};

pub mod optimiser {
    use crate::nn::ExecutionContext;
    use bullet_core::optimiser::{self, clip, decay, radam, utils::Placement, OptimiserState};

    type ClipAndDecay<T> = clip::WeightClipping<decay::WeightDecay<T>>;

    pub type AdamWOptimiser = optimiser::adam::AdamW<ExecutionContext>;
    pub type RAdamOptimiser = ClipAndDecay<radam::RAdam<ExecutionContext>>;
    pub type RangerOptimiser = optimiser::ranger::Ranger<ExecutionContext>;
    pub use optimiser::{adam::AdamWParams, ranger::RangerParams, Optimiser};

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

    #[derive(Default)]
    pub struct Ranger;
    impl OptimiserType for Ranger {
        type Optimiser = RangerOptimiser;
    }

    #[derive(Clone, Copy, Debug)]
    pub struct RAdamParams {
        pub decay: f32,
        pub beta1: f32,
        pub beta2: f32,
        pub min_weight: f32,
        pub max_weight: f32,
    }

    type ClipAndDecayParams<T> = clip::WeightClippingParams<decay::WeightDecayParams<T>>;

    impl From<RAdamParams> for ClipAndDecayParams<radam::RAdamParams> {
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
