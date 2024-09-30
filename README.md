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

### Backends:

Building `bullet` requires a C++ compiler - it is recommend to use `clang`.
- If on Windows, get it directly from LLVM github releases
- You may need to specify the environment variable `CXX=clang++`

#### CUDA
The default backend. You will need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

#### HIP

> [!NOTE]
> If you are on Windows, you must also add `%HIP_PATH%\bin\` to the PATH variable in your system environment variables.

> [!WARNING]  
> The HIP backend is not *officially* supported on Linux (due to unresolved issues with annoying platform dependent stuff), but it has been made to work by a couple of users with some minor edits.

For users with AMD GPUs. To compile to target HIP you need to enable the `hip` feature. You will need to install the [HIP SDK](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html).