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

### Currently Supported Backends:
#### Default
Reference CPU backend. It is suitable for training small networks or various utilities, such as loading nets to requantise them or test their output on specific positions.

> [!WARNING]
> Not intended for serious training use. If you need to train on CPU, use the [legacy](https://github.com/jw1912/bullet/tree/legacy) branch.

#### CUDA
The "first class" supported backend. To compile to target CUDA you need to enable the `cuda` feature,
as demonstrated in the [wiki](https://github.com/jw1912/bullet/wiki/2.-Getting-Started-with-bullet). You will need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), and have an available C++ compiler.

> [!NOTE]
> If you are on Windows, it is recommended to use clang, direct from LLVM github releases, for the C++ compiler.

#### HIP
Mainly directed toward users with AMD GPUs. To compile to target HIP you need to enable the `hip` feature,
as demonstrated in the [wiki](https://github.com/jw1912/bullet/wiki/2.-Getting-Started-with-bullet). You will need to install the [HIP SDK](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html).

> [!NOTE]
> If you are on Windows, you must also add `%HIP_PATH%\bin\` to the PATH variable in your system environment variables.

> [!WARNING]  
> Due to what appears to be a bug in RoCM, some tests will sometimes fail due to missed synchronisation between device and host in a multithreaded context. As the trainer only calls kernels from one thread, this should not be an issue in training.

> [!WARNING]  
> The HIP backend is not *officially* supported on Linux (due to unresolved issues with annoying platform dependent stuff), but it has been made to work by a couple of users with some minor edits.