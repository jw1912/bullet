/// Contains `Trainer`, the default implementation
/// of `NetworkTrainer`, for training value networks.
#[deprecated]
pub mod default;

/// Contains common game related code used in both policy
/// and value network traiing (e.g. sparse input types).
pub mod game;

/// Contains the `NetworkTrainer` trait and
/// associated structs for its use
pub mod trainer;

/// Contains `ValueTrainer`, the new and improved
/// way to train value networks.
pub mod value;

// TODO: Remove these re-exports as they are exported in the `nn` module
pub use bullet_core::graph::builder::{Activation, Shape};
pub use nn::ExecutionContext;

// TODO: Remove these re-exports as they are exported in the `trainer` module
pub use trainer::{
    schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
    settings::LocalSettings,
};

/// Re-export of `bullet_core`
pub use bullet_core as core;

/// Contains the Graph API, by which neural networks are created with
/// `NetworkBuilder`, and then compiled into an executable `Graph`
pub mod nn;

#[cfg(all(feature = "hip-cuda", any(feature = "cpu", feature = "cuda")))]
compile_error!(
    "In order to use a non-HIP backend, you must pass the `--no-default-features` flag.
If running an example, this would be
    cargo r -r --example <example name> --features <your feature> --no-default-features
If using bullet as a crate, it is instead
    bullet_lib = { .. other stuff here .. , default-features = false }"
);
