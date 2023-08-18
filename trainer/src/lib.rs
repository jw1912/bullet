pub mod arch;
pub mod activation;
pub mod trainer;

/// Size of hidden layer.
pub const HIDDEN_SIZE: usize = 256;

/// Activation Function:
///   - ReLU
///   - CReLU
pub type ActivationUsed = activation::ReLU;
