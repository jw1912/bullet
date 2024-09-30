<div align="center">

# bullet

</div>

NN trainer, generally used for training NNUE-style networks for chess engines.

### Usage

Most people simply clone the repo and edit one of the [examples](/examples) to their taste.

Alternatively, import the crate with
```toml
bullet = { package = "bullet_lib", version = "1.0.0", features = ["cuda"] }
```

Check out the [wiki](https://github.com/jw1912/bullet/wiki/2.-Getting-Started-with-bullet) and [examples](/examples) to see how to use the crate.

### Utilities

You can build `bullet-utils` with `cargo b -r --package bullet-utils`, to do the following:
- Convert Data
- Interleave Multiple Data Files
- Shuffle Data Files
- Validate Data Files

Use `./target/release/bullet-utils[.exe] help` to see specific usage.

### Backends

#### General
Building `bullet` requires a C++ compiler (which will be invoked by `nvcc` or `hipcc`).
- On Windows, this should be `cl.exe` (requires Visual Studio to be installed)
- On Linux, it is recommended to use `clang`
    - You may need to specify the environment variable `CXX` or `HOST_CXX` with the compiler name

#### CUDA
The default backend.
- Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- CUDA version >=12.2 is required.
- The `CUDA_PATH` environment variable must be set to the CUDA install location (should contain the `bin`, `lib` and `include` directories)
- The system `PATH` should contain `%CUDA_PATH%\bin` (or equivalent for Linux)

#### HIP
For users with AMD GPUs.
- Enable the `hip` feature
- Install the [HIP SDK](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html)
- The `HIP_PATH` environment variable must be set to the HIP install location (should contain the `bin`, `lib` and `include` directories)
- The system `PATH` should contain `%HIP_PATH%\bin` (or equivalent for Linux)
