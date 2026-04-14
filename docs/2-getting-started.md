# Getting Started

### Installing Rust

Install Rust via [rustup](https://www.rust-lang.org/tools/install) (this is the official way to install rust).

### General Usage

You can use `bullet` as a crate:
```toml
bullet = { git = "https://github.com/jw1912/bullet", package = "bullet_lib" }
```
or by editing and running one of the [examples](../examples):
```
cargo r -r --example <example name>
```

A basic inference example is included in [examples/simple](../examples/simple.rs), and if you've never
trained an NNUE before it is recommended to start with an architecture and training schedule similar to it.

### Utilities

You can build `bullet-utils` with `cargo b -r --package bullet-utils`, to do the following:
- Convert between data formats
- Interleave multiple data files
- Shuffle data files
- Validate data files

Use `./target/release/bullet-utils[.exe] help` to see specific usage.

This does **not** require CUDA or HIP.

### Backends

#### CUDA

For users with NVIDIA GPUs.

- Enable the `cuda` feature
- Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
    - Recommended to use as recent a version as possible
    - If the toolkit version is too old you should receive a relatively clear error, either at compile time via a linker error or at runtime
- The `CUDA_PATH` environment variable must be set to the CUDA install location (should contain the `bin`, `lib` and `include` directories)

#### ROCm

For users with AMD GPUs.

- Enable the `rocm` feature
- Install the [HIP SDK](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html)
- The `HIP_PATH` environment variable must be set to the HIP install location (should contain the `bin`, `lib` and `include` directories)
- You will probably need to specify the `GCN_ARCH_NAME` environment variable - which you should be able to find using `rocminfo` on Linux, or `hipinfo` on Windows
