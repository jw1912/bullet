<div align="center">

# bullet

</div>

A CUDA-powered Neural Network Trainer, used to train NNUE-style networks for [akimbo](https://github.com/jw1912/akimbo).

Also used by many other engines, including but not limited to:
- [Alexandria](https://github.com/PGG106/Alexandria)
- [Altair](https://github.com/Alex2262/AltairChessEngine)
- [Carp](https://github.com/dede1751/carp)
- [Midnight](https://github.com/archishou/MidnightChessEngine)
- [Obsidian](https://github.com/gab8192/Obsidian)
- [Stormphrax](https://github.com/Ciekce/Stormphrax)
- [Willow](https://github.com/Adam-Kulju/Willow)

### Usage

Check out the [wiki](https://github.com/jw1912/bullet/wiki/1.-Getting-Started) and [examples](/examples) to see how to use the crate.

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
as demonstrated in the [wiki](https://github.com/jw1912/bullet/wiki/1.-Getting-Started).

#### BLAS
A (very) minor speedup for CPU training. Currently tested on Windows with [OpenBLAS precompiled packages](https://github.com/OpenMathLib/OpenBLAS/releases).
