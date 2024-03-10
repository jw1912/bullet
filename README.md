<div align="center">

# bullet

</div>

A CUDA-powered Neural Network Trainer, used to train NNUE-style networks for [akimbo](https://github.com/jw1912/akimbo).

Also used by many other engines, including:
- [Alexandria](https://github.com/PGG106/Alexandria)
- [Altair](https://github.com/Alex2262/AltairChessEngine)
- [Carp](https://github.com/dede1751/carp)
- [Midnight](https://github.com/archishou/MidnightChessEngine)
- [Obsidian](https://github.com/gab8192/Obsidian)
- [Stormphrax](https://github.com/Ciekce/Stormphrax)
- [Willow](https://github.com/Adam-Kulju/Willow)
- [Viridithas](https://github.com/cosmobobak/viridithas)

### Usage

Check out the [wiki](https://github.com/jw1912/bullet/wiki/2.-Getting-Started-with-bullet) and [examples](/examples) to see how to use the crate.

### Utilities

You can build `bullet-utils` with `cargo b -r --package bullet-utils`, to do the following:
- Convert Data
- Interleave Multiple Data Files
- Shuffle Data Files
- Validate Data Files

Use `./target/release/bullet-utils[.exe] help` to see specific usage.

### Currently Supported Games:
- Chess
- Ataxx

Raise an issue for support of a new game.

### Currently Supported Backends:
#### Default
CPU backend **not intended for serious training use**. It is suitable for training small networks or various utilities,
such as loading nets to requantise them or test their output on specific positions.

#### CUDA
The "first class" supported backend. To compile to target CUDA you need to enable the `cuda` feature,
as demonstrated in the [wiki](https://github.com/jw1912/bullet/wiki/2.-Getting-Started-with-bullet).

#### BLAS
A (very) minor speedup for CPU training. Currently tested on Windows with [OpenBLAS precompiled packages](https://github.com/OpenMathLib/OpenBLAS/releases).
