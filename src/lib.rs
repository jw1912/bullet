pub mod network;
pub mod position;
pub mod rng;
pub mod trainer;
pub mod util;

/// Size of hidden layer.
pub const HIDDEN_SIZE: usize = 32;

/// Activation Function:
///   - ReLU
///   - CReLU
///   - SCReLU
pub type ActivationUsed = network::activation::CReLU;

/// Optimiser:
///   - Adam
///   - AdamW
pub type OptimiserUsed = trainer::optimiser::AdamW;
