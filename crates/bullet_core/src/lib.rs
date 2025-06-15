/// Contains the backend, mainly the `Device` and `DeviceBuffer` APIs.
pub mod backend;
/// Contains the raw graph construction and execution API.
pub mod graph;
/// Contains the `Optimiser` struct and `OptimiserState` trait,
/// as well as some provided optimisers.
pub mod optimiser;
/// Contains the `Trainer` struct and `TrainerState` trait.
pub mod trainer;
