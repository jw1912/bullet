pub mod activation;
pub mod data;
pub mod inputs;
pub mod rng;
pub mod util;

// Using CUDA:
// At the moment this will hard lock you into using
// CReLU Activation and the AdamW Optimiser.

/// Binary data type used
///  - ChessBoard
pub type Data = data::ChessBoard;

/// Input format
///  - Chess768  (recommended)
///  - HalfKA
pub type Input = inputs::Chess768;

/// Size of hidden layer.
pub const HIDDEN: usize = 768;

/// Activation function:
///   - ReLU
///   - CReLU    (recommended)
///   - SCReLU
pub type Activation = activation::CReLU;
