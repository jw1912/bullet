/// Re-export of `acyclib`
pub use acyclib;

/// Contains common game related code used in both policy
/// and value network traiing (e.g. sparse input types).
pub mod game;

/// Contains the Graph API, by which neural networks are created with
/// `NetworkBuilder`, and then compiled into an executable `Graph`
pub mod nn;

/// Contains the `NetworkTrainer` trait and
/// associated structs for its use
pub mod trainer;

/// Contains `ValueTrainer`, the new and improved
/// way to train value networks.
pub mod value;

// TODO: Remove these re-exports as they are exported in the `trainer` module
pub use trainer::{
    schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
    settings::LocalSettings,
};
