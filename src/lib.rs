pub mod cuda;
pub mod data;
pub mod network;
pub mod rng;
pub mod trainer;
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
pub type Input = network::inputs::Chess768;

/// Size of hidden layer.
pub const HIDDEN: usize = 8;

/// Activation function:
///   - ReLU
///   - CReLU    (recommended)
///   - SCReLU
pub type Activation = network::activation::CReLU;

/// Optimiser:
///   - Adam
///   - AdamW    (recommended)
pub type Optimiser = trainer::optimiser::AdamW;
