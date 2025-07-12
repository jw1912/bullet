/// Token single-threaded CPU backend, for verifying correctness of other backends.
pub mod cpu;
/// Contains `Device` and `DeviceBuffer` APIs.
pub mod device;
/// Contains the raw graph construction and execution API.
pub mod graph;
/// Contains the `Optimiser` struct and `OptimiserState` trait,
/// as well as some provided optimisers.
pub mod optimiser;
/// Contains the `Trainer` struct and `TrainerState` trait.
pub mod trainer;
