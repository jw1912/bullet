# Bullet Documentation

This is the documentation for the basic usage of `bullet_lib`.

### NNUE Guide

Note that this guide is specifically for NNUE training.

1. [NNUE Basics](1-basics.md)
    - [Simple Feed-Forward Network](1-basics.md#simple-feed-forward-network)
    - [Perspective Networks](1-basics.md#perspective-networks)
    - [Beginner Traps](1-basics.md#beginner-traps)
    - [Good NNUE Resources](1-basics.md#good-nnue-resources)
2. [Getting Started](2-getting-started.md)
    - [General Usage](2-getting-started.md#general-usage)
    - [Utilities](2-getting-started.md#utilities)
    - [Backends](2-getting-started.md#backends)
3. [Training Data](3-data.md)
    - [General Workflow](3-data.md#general-workflow)
    - [Custom Data Types](3-data.md#custom-data-types)
    - [Custom Data Loading](3-data.md#custom-data-loading)
4. [Saved Networks](4-saved-networks.md)
    - [Checkpoint Layout](4-saved-networks.md#checkpoint-layout)
    - [Loading Checkpoints](4-saved-networks.md#loading-checkpoints)
    - [Network Layout with `TrainerBuilder`](4-saved-networks.md#network-layout-with-trainerbuilder)

### Crates & Versioning

#### [bullet_core](crates/bullet_core)

- Follows SemVer with crates.io releases
- Contains the `Device` and `DeviceBuffer` traits, which are used to define a backend
- Network graph construction, execution and autodiff generic over backends

#### [bullet_cuda_backend](crates/bullet_cuda_backend)

- Follows SemVer with crates.io releases
- New CUDA backend
- Not currently in a useable state

#### [bullet_hip_backend](crates/bullet_hip_backend)

- Follows SemVer with crates.io releases
- Contains both the HIP and CUDA backends

#### [bullet_lib](crates/bullet_lib)

- Does not follow any particular versioning
- API is sometimes subject to breaking changes
- `NetworkTrainer` traits wrap the core graph API, providing a training loop with data loading performed asynchronously from device calculations
- Contains `Trainer`, which implements `NetworkTrainer` for value network training
- Contains `TrainerBuilder`, which streamlines the process of constructing a `Trainer` for the most common network architectures
- Lots of NNUE and chess engine specific tooling
    - Input feature types
    - Output buckets
    - Data formats

#### [bullet-utils](crates/bullet_utils)

- Does not follow any particular versioning
- Is a CLI program with various chess related utilities
- Data validation
- Converting between data file types
- Shuffling and interleaving data files
