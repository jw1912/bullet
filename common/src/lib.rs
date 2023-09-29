pub mod activation;
pub mod data;
pub mod inputs;
pub mod outputs;
pub mod rng;
pub mod util;

/// Binary data type used
///  - ChessBoard
pub type Data = data::ChessBoard;

/// Input format
///  - Chess768  (recommended)
///  - HalfKA
///  - MirroredHalfKA
pub type Input = inputs::HalfKA;

/// Size of hidden layer.
pub const HIDDEN: usize = 32;

/// Activation function:
///   - ReLU
///   - CReLU    (recommended)
///   - SCReLU
///   - FastSCReLU
pub type Activation = activation::CReLU;

/// Output Buckets:
///  - Single    (recommended)
///  - MaterialCount<Buckets>
pub type OutputBucket = outputs::Single;
