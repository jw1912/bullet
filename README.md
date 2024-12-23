<div align="center">

# bullet-legacy

</div>

This is the legacy branch of bullet, intended for fast CPU training of chess networks with architecture `(INPUTS -> HL_SIZE)x2 -> 1xOUTPUT_BUCKETS`.

All the options for training are found in `src/main.rs`, and you run the trainer with `cargo r -r`.

The docs from the main branch of bullet still apply and you should use `bullet-utils` from the main branch for dealing with data,
however checkpoints are not interchangeable between main and legacy (at present).

Restrictions that come with legacy:
- Completely restricted network architecture
- Can only use `DirectSequentialDataLoader` and only use `ChessBoard` as the input type
    - This is what is generally referred to as "bulletformat", files are a series of 32-byte records
- No support for validation data
- No support for dispatching cutechess/fastchess games to test throughout training
- `InputType` and `OutputBuckets` are less flexible
- There are probably even more that I have forgotten
