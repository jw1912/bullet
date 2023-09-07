pub mod data;
pub mod network;
pub mod rng;
pub mod trainer;
pub mod util;

/// Binary data type used
///  - ChessBoard
pub type Data = data::ChessBoard;

/// Input format
///  - Chess768
///  - HalfKA
pub type Input = network::inputs::Chess768;

/// Size of hidden layer.
pub const HIDDEN: usize = 256;

/// Activation function:
///   - ReLU
///   - CReLU
///   - SCReLU
pub type Activation = network::activation::CReLU;

/// Optimiser:
///   - Adam
///   - AdamW
pub type Optimiser = trainer::optimiser::AdamW;
