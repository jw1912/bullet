# Getting Started

### Installing Rust

Install Rust via [rustup](https://www.rust-lang.org/tools/install) (this is the official way to install rust).

### General Usage

You can use `bullet` as a crate:
```toml
bullet = { git = "https://github.com/jw1912/bullet" }
```
or by editing and running one of the [examples](https://github.com/jw1912/bullet/tree/main/examples):
```
cargo r -r --example <example name>
```

A basic inference example is included in [examples/simple](https://github.com/jw1912/bullet/tree/main/examples/simple.rs), and if you've never
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

#### General
Building `bullet` requires a C++ compiler (which will be invoked by `nvcc` or `hipcc`).
- On Windows, this should be `cl.exe` (requires Visual Studio to be installed)
- On Linux, it is recommended to use `clang`
    - You may need to specify the environment variable `CXX` or `HOST_CXX` with the compiler name

#### CUDA
The default backend when compiling `bullet_lib`.
- You **should not** enable any features to use this backend
- Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- Recommended CUDA version >=12.2 (lower versions can work)
- The `CUDA_PATH` environment variable must be set to the CUDA install location (should contain the `bin`, `lib` and `include` directories)
- The system `PATH` should contain `%CUDA_PATH%\bin` (or equivalent for Linux)

#### HIP
For users with AMD GPUs.
- Enable the `hip` feature
- Install the [HIP SDK](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html)
- The `HIP_PATH` environment variable must be set to the HIP install location (should contain the `bin`, `lib` and `include` directories)
- The system `PATH` should contain `%HIP_PATH%\bin` (or equivalent for Linux)
- On Linux, you will need to specify the `GCN_ARCH_NAME` environment variable, which you should be able to find using `rocminfo`.

#### CPU
If you need to train on CPU you can use the [legacy branch](https://github.com/jw1912/bullet/tree/legacy).
