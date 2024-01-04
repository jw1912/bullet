<div align="center">

# bullet

</div>

A CUDA-powered Neural Network Trainer, used to train NNUE-style networks for [akimbo](https://github.com/jw1912/akimbo).

Also used by a number of other engines, including:
- [Alexandria](https://github.com/PGG106/Alexandria)
- [Altair](https://github.com/Alex2262/AltairChessEngine)
- [Carp](https://github.com/dede1751/carp)
- [Midnight](https://github.com/archishou/MidnightChessEngine)
- [Obsidian](https://github.com/gab8192/Obsidian)
- [Stormphrax](https://github.com/Ciekce/Stormphrax)
- [Willow](https://github.com/Adam-Kulju/Willow)

## About

Used exclusively to train architectures of the form `(SparseInput -> N)x2 -> MoreHiddenLayers -> 1`.

If you need to train on CPU, you can use the [legacy branch](https://github.com/jw1912/bullet/tree/legacy).

Currently Supported Games:
- Chess
- Ataxx

Raise an issue for support of a new game.

### Usage

Check out the [examples](/examples) to see how to use the crate - and if you don't want to use bullet
as a crate, you can edit one of them and run with `cargo r -r --example <example name (without .rs)>`.

A basic inference example is included in [examples/akimbo-main](/examples/akimbo-main.rs).

### Saved Networks

When a checkpoint is saved to a directory `<out_dir>/<checkpoint_name>`, it will contain
- `params.bin`, the raw floating point (`f32`) parameters of the network.
- `momentum.bin`, used by the optimiser in training.
- `velocity.bin`, used by the optimiser in training.

If quantisation has been specified then it will also contain `<checkpoint_name>.bin`, which
is the quantised network parameters, each weight being stored in an `i16` - if quantisation
fails (due to overflowing the i16 limits), then it will not save the quantised network and
inform the user to quantise manually (using `params.bin`), but training will be otherwise unaffected.

In each case, the format of these files is `(layer 1 weights)(layer 1 biases)(layer 2 weights)...` stored
contiguously, with the weights matrices being stored column-major.

## Data

The trainer uses its own binary data format for each game.

The specifications for the data formats are found in the [bulletformat](https://github.com/jw1912/bulletformat) crate.

Additionally, each type implements `from_raw` which is recommended for use if your engine is written in Rust (or you don't
mind FFI).

All data types at present are 32 bytes, so you can use [marlinflow-utils](https://github.com/jnlt3/marlinflow) to shuffle
and interleave files.

### Ataxx

You can convert text format, where
- each line is of the form `<FEN> | <score> | <result>`
- `FEN` has 'x'/'r', 'o'/'b' and '-' for red, blue and gaps/blockers, respectively, in the same format as FEN for chess
- `score` is red relative and an integer
- `result` is red relative and of the form `1.0` for win, `0.5` for draw, `0.0` for loss

by using the command
```
cargo r -r --package bullet-utils --bin convertataxx <input file path> <output file path>
```

### Chess

You can convert a [Marlinformat](https://github.com/jnlt3/marlinflow) file by running
```
cargo r -r --package bullet-utils --bin convertmf <input file path> <output file path> <threads>
```
it is up to the user to provide a valid Marlinformat file.

Additionally, you can convert legacy text format as in Marlinflow, where
- each line is of the form `<FEN> | <score> | <result>`
- `score` is white relative and in centipawns
- `result` is white relative and of the form `1.0` for win, `0.5` for draw, `0.0` for loss

by using the command
```
cargo r -r --package bullet-utils --bin convert <input file path> <output file path>
```
