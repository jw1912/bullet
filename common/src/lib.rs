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
///  - ChessBuckets
pub type Input = inputs::Chess768;

/// Size of hidden layer.
pub const HIDDEN: usize = 256;

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

/// Applicable only with `Input` option
/// "CustomBuckets", it is indexed from white POV,
/// so index 0 corresponds to A1, 7 corresponds to H1,
/// 56 corresponds to A8 and 63 corresponds to H8.
/// With `N` buckets, the values in this array need to be
/// in the set {0, 1, ..., N - 1}.
pub const BUCKETS: [usize; 64] = [
    0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 1, 1,
    2, 2, 2, 2, 3, 3, 3, 3,
    2, 2, 2, 2, 3, 3, 3, 3,
    2, 2, 2, 2, 3, 3, 3, 3,
    2, 2, 2, 2, 3, 3, 3, 3,
    2, 2, 2, 2, 3, 3, 3, 3,
    2, 2, 2, 2, 3, 3, 3, 3,
];
