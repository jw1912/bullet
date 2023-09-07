pub mod network;
pub mod data;
pub mod rng;
pub mod trainer;
pub mod util;

/// Binary data type used
pub type Data = data::MarlinFormat;

/// Input format
pub type Input = network::inputs::Chess768;

/// Size of hidden layer.
pub const HIDDEN: usize = 32;

/// Activation function:
///   - ReLU
///   - CReLU
///   - SCReLU
pub type ActivationUsed = network::activation::CReLU;

/// Optimiser:
///   - Adam
///   - AdamW
pub type OptimiserUsed = trainer::optimiser::AdamW;
