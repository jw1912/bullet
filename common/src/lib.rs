pub mod activation;
pub mod data;
pub mod inputs;
pub mod rng;
pub mod util;

/// Binary data type used
///  - ChessBoard
pub type Data = data::ChessBoard;

/// Input format
///  - Chess768  (recommended)
///  - HalfKA
pub type Input = inputs::Chess768;

/// Size of hidden layer.
pub const HIDDEN: usize = 256;

/// Activation function:
///   - ReLU
///   - CReLU    (recommended)
///   - SCReLU
///   - FastSCReLU
pub type Activation = activation::CReLU;
