pub use acyclib::{
    device::tensor::Shape,
    graph::{
        Node,
        builder::{Affine, GraphBuilder as NetworkBuilder, GraphBuilderNode as NetworkBuilderNode, InitSettings},
    },
};

#[cfg(any(feature = "multigpu", feature = "cpu"))]
pub type Graph = acyclib::graph::multi::MultiDeviceGraph<ExecutionContext>;

#[cfg(not(any(feature = "multigpu", feature = "cpu")))]
pub type Graph = acyclib::graph::Graph<ExecutionContext>;

#[cfg(all(feature = "cpu", not(feature = "cuda")))]
pub use acyclib::device::cpu::{CpuError as DeviceError, CpuMarker as BackendMarker, CpuThread as ExecutionContext};

#[cfg(all(any(feature = "hip", feature = "hip-cuda"), not(feature = "cpu"), not(feature = "cuda")))]
pub use bullet_hip_backend::{DeviceError, ExecutionContext, HipMarker as BackendMarker};

#[cfg(feature = "cuda")]
pub use bullet_cuda_backend::{CudaDevice as ExecutionContext, CudaError as DeviceError, CudaMarker as BackendMarker};

pub mod optimiser {
    use crate::nn::ExecutionContext;
    use acyclib::trainer::optimiser::{self, OptimiserState, radam};

    pub type AdamWOptimiser = optimiser::adam::AdamW<ExecutionContext>;
    pub type RAdamOptimiser = radam::RAdam<ExecutionContext>;
    pub type RangerOptimiser = optimiser::ranger::Ranger<ExecutionContext>;
    pub use optimiser::{Optimiser, adam::AdamWParams, ranger::RangerParams};

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

    impl From<RAdamParams> for radam::RAdamParams {
        fn from(value: RAdamParams) -> Self {
            radam::RAdamParams {
                beta1: value.beta1,
                beta2: value.beta2,
                n_sma_threshold: 5.0,
                decay: value.decay,
                clip: Some((value.min_weight, value.max_weight)),
            }
        }
    }
}
