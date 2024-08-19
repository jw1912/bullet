<div align="center">

# bullet

</div>

A CUDA/CPU NN Trainer, used to train NNUE-style networks for [akimbo](https://github.com/jw1912/akimbo).

Also used by many other engines, including:
- [Alexandria](https://github.com/PGG106/Alexandria)
- [Altair](https://github.com/Alex2262/AltairChessEngine)
- [Carp](https://github.com/dede1751/carp)
- [Midnight](https://github.com/archishou/MidnightChessEngine)
- [Obsidian](https://github.com/gab8192/Obsidian)
- [Stormphrax](https://github.com/Ciekce/Stormphrax)
- [Willow](https://github.com/Adam-Kulju/Willow)
- [Viridithas](https://github.com/cosmobobak/viridithas)

### Currently Supported Games:
- Chess
- Ataxx

Raise an issue for support of a new game.

### Usage

Import the crate with
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
> Not intended for serious training use.

#### CUDA
The "first class" supported backend. To compile to target CUDA you need to enable the `cuda` feature,
as demonstrated in the [wiki](https://github.com/jw1912/bullet/wiki/2.-Getting-Started-with-bullet).

> [!NOTE]
> If you are on Windows, it is recommended to use clang, direct from LLVM github releases.

#### HIP
Mainly directed toward users with AMD GPUs. To compile to target HIP you need to enable the `hip` feature,
as demonstrated in the [wiki](https://github.com/jw1912/bullet/wiki/2.-Getting-Started-with-bullet). You will need to install the [HIP SDK](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html).

> [!NOTE]
> If you are on Windows, you must also add `%HIP_PATH%\bin\` to the PATH variable in your system environment variables.

> [!WARNING]  
> Due to what appears to be a bug in RoCM, some tests will sometimes fail due to missed synchronisation between device and host in a multithreaded context. As the trainer only calls kernels from one thread, this should not be an issue in training.

> [!WARNING]  
> The HIP backend is not *officially* supported on Linux (due to unresolved issues with annoying platform dependent stuff), but it has been made to work by a couple of users with some minor edits.