# Training Data

## General Workflow

1. Store your data in some format (heavily recommended to use a binpack-like format)
2. Convert data files from your this format to a data format that `Trainer` can ingest if needed
3. Shuffle the individual converted files
4. Interleave the shuffled files

## Builtin Data Loaders

You can easily write a dataloader for your own format if you wish, but bullet already contains loaders for the most common formats.

### Binpacks

Stockfish, Monty and Viridithas format "binpacks" can be loaded using `SfBinpackLoader`, `MontyBinpackLoader` and `ViriBinpackLoader` respectively.
Binpacks stores entire games contiguously to achieve great compression and hence need a filter function to be passed to remove e.g. noisy positions.

I would recommend using Viriformat binpacks as they are the most commonly used amongst people who generate their own data (and thus there are reference
implementations in many programming languages, as well as many utilites available).

### ChessBoard aka "bulletformat"

This is a simple and fast-to-load data format that can be loaded with `DirectSequentialDataLoader`.
It is suitable for training small networks.
However it is recommended to generate and store data in a binpack-like format, and only convert to this format if bottlenecked by data loading speed.
It throws away information such as the side-to-move (the record is stored stm-relative), halfmove counter, castling rights, etc.
The `bullet-utils` binary contains utilities for shuffling and interleaving these data files, as well as converting from some other data formats.

In particular, you can convert to this data type from a text file that contains a list of data points in the following form:
- each line is of the form `<FEN> | <score> | <result>`
- `score` is white relative and in centipawns
- `result` is white relative and of the form `1.0` for win, `0.5` for draw, `0.0` for loss
