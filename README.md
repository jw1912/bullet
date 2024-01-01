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

Currently Supported Games:
- Chess
- Ataxx

Raise an issue for support of a new game.

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
cargo r -r --bin convertataxx <input file path> <output file path>
```

### Chess

You can convert a [Marlinformat](https://github.com/jnlt3/marlinflow) file by running
```
cargo r -r --bin convertmf <input file path> <output file path> <threads>
```
it is up to the user to provide a valid Marlinformat file.

Additionally, you can convert legacy text format as in Marlinflow, where
- each line is of the form `<FEN> | <score> | <result>`
- `score` is white relative and in centipawns
- `result` is white relative and of the form `1.0` for win, `0.5` for draw, `0.0` for loss

by using the command
```
cargo r -r --bin convert <input file path> <output file path>
```
