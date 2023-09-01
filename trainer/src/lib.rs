pub mod activation;
pub mod arch;
pub mod optimiser;
pub mod rng;
pub mod scheduler;
pub mod trainer;

/// Size of hidden layer.
pub const HIDDEN_SIZE: usize = 32;

/// Activation Function:
///   - ReLU
///   - CReLU
///   - SCReLU
pub type ActivationUsed = activation::ReLU;

/// Optimiser:
///   - Adam
///   - AdamW
pub type OptimiserUsed = optimiser::AdamW;
