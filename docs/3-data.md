# Training Data

Compile `bullet-utils` with `cargo b -r --package bullet-utils`, run with `./target/release/bullet-utils[.exe] help` to see usage instructions.
These utilities can be used for shuffling and interleaving certain data type files.

## General Workflow

1. Store your data in some format (heavily recommended to use a binpack-like format) 
2. Convert data files from your this format to a data format that `Trainer` can ingest if needed
3. Shuffle the individual converted files
4. Interleave the shuffled files

## Provided Data Types

The specifications for the data formats are found in the [`bulletformat`](https://github.com/jw1912/bulletformat) crate.

Each provided binary data type implements `from_raw` which is recommended for use in datagen if your engine is written in Rust
(or you don't mind FFI).

Generally speaking, other than getting data in the required format via the methods described below, you won't need to think
about them at all unless you want to do custom inputs.

### ChessBoard aka "bulletformat"

This data type can be loaded with `DirectSequentialDataLoader`.

This is the standard data format for chess training, it is a fixed 32-byte record, with some spare bytes for storing custom info.
It throws away information such as the side-to-move (the record is stored stm-relative), halfmove counter, castling rights, etc.

You can convert from a [Marlinformat](https://github.com/jnlt3/marlinflow) or text format file using `bullet-utils`.

Text Format:
- each line is of the form `<FEN> | <score> | <result>`
- `score` is white relative and in centipawns
- `result` is white relative and of the form `1.0` for win, `0.5` for draw, `0.0` for loss


### Stockfish & Monty Binpacks

These types can be loaded with `SfBinpackLoader` and `MontyBinpackLoader` respectively.
There are utilities for interleaving Monty binpacks in `bullet-utils`.
Stockfish contains tools for interleaving its own binpack format.

## Custom Data Types

It is relatively easy to support custom data types for use in the default `Trainer` whereby most of the complexity is abstracted away.
The process is as follows:

1. Write a custom data type `CustomDataType`
2. Implement `LoadableDataType` for `CustomDataType`
3. Write an input type `CustomInputs` by implementing `SparseInputType` with `RequiredDataType = CustomDataType`
for it, which is the method for extracting inputs from `CustomDataType`
4. Implement a corresponding data loader `CustomDataLoader` for the custom data type that handles reading the data from a file
by implementing `DataLoader<CustomDataType>` for it
    - If the data type is a fixed-size byte record that can be safely transmuted from any `[u8; std::mem::size_of::<CustomDataType>()]`,
    then you can `unsafe impl CanBeDirectlySequentiallyLoaded for CustomDataType` and use `DirectSequentialDataLoader`
