/// Contains the backend, mainly the `Device` and `DeviceBuffer` APIs.
pub mod backend;
/// The single-threaded CPU backend.
///pub mod cpu;
/// Contains the raw graph construction and execution API.
pub mod graph;
/// Contains the `Optimiser` struct and `OptimiserState` trait,
/// as well as some provided optimisers.
pub mod optimiser;
