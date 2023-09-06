pub mod network;
pub mod position;
pub mod rng;
pub mod trainer;
pub mod util;

pub type Input = network::inputs::Chess768;

/// Size of hidden layer.
pub const HIDDEN: usize = 32;

/// Activation Function:
///   - ReLU
///   - CReLU
///   - SCReLU
pub type ActivationUsed = network::activation::CReLU;

/// Optimiser:
///   - Adam
///   - AdamW
pub type OptimiserUsed = trainer::optimiser::AdamW;
