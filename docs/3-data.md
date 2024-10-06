# Training Data

Compile `bullet-utils` with `cargo b -r --package bullet-utils`, run with `./target/release/bullet-utils[.exe] help` to see usage instructions.

Bullet generally functions by ingesting **binary** data formats (not text), so if you have a text format, you will need to convert it.

## General Workflow

1. Convert data files from your own format to `BulletFormat` compatible data
2. Shuffle the individual converted files
3. Interleave the shuffled files

## Provided Data Types

The specifications for the data formats are found in the [`bulletformat`](https://github.com/jw1912/bulletformat) crate.

Each provided binary data type implements `from_raw` which is recommended for use in datagen if your engine is written in Rust
(or you don't mind FFI).

Generally speaking, other than getting data in the required format via the methods described below, you won't need to think
about them at all unless you want to do custom inputs.

### ChessBoard

This is the standard data format for chess training, it is a fixed 32-byte record, with some spare bytes for storing custom info.

You can convert a [Marlinformat](https://github.com/jnlt3/marlinflow) file.

Additionally, you can convert text format, where
- each line is of the form `<FEN> | <score> | <result>`
- `score` is white relative and in centipawns
- `result` is white relative and of the form `1.0` for win, `0.5` for draw, `0.0` for loss

### AtaxxBoard

This is the standard data format for ataxx training.

You can convert text format, where
- each line is of the form `<FEN> | <score> | <result>`
- `FEN` has 'x'/'r', 'o'/'b' and '-' for red, blue and gaps/blockers, respectively, in the same format as FEN for chess
- `score` is red relative and an integer
- `result` is red relative and of the form `1.0` for win, `0.5` for draw, `0.0` for loss

## Custom Data Types

Any type that implements the `BulletFormat` trait can be used in the trainer.
For example, `MarlinFormat` is implemented in the `bulletformat` crate, so it can be easily used if preferred.

## Custom Data Loading

Your required data type `T: BulletFormat` can be loaded from any file format (e.g. text or binpack) by proiding a custom
dataloader that implements the `DataLoader<T>` trait.
