<div align="center">

# bullet

</div>

A work-in-progress NNUE Trainer, used to train [akimbo](https://github.com/JacquesRW/akimbo)'s networks.

It currently supports architectures of the form `Input -> Nx2 -> 1`, and can train on CPU with any number of threads.

Supported input formats:
- `Chess768`, the classic chess board of features `(colour, piece, square)`.
- `HalfKA`, chess board of features `(friendly king square, colour, piece, square)`

To learn how it works, read the [wiki](wiki.md).

## Usage

### Data

The trainer uses its own binary data format.

You can convert a [Marlinformat](https://github.com/jnlt3/marlinflow) file by running
```
cargo r -r --bin convertmf <input file path> <output file path>
```
it is up to the user to provide a valid Marlinformat file, as well as shuffling the data beforehand.

Additionally, you can convert a text file with each line of the form `<FEN> <score> <result>`, where
- `FEN` string must be complete (with halfmove and fullmove numbers)
- `score` is white relative and in centipawns
- `result` is white relative and of the form `[1.0]` for win, `[0.5]` for draw, `[0.0]` for loss

by using the command
```
cargo r -r --bin convert <input file path> <output file path>
```

### Training

General architecture settings, that must be known at compile time, are found in [`src/lib.rs`](src/lib.rs).
It is like this because of Rust's limitations when it comes to const code.

After settings those as you please, you can run the trainer using the `run.py` script, and use
```
python3 run.py --help
```
to get a full description of all options.

A sample usage is
```
python3 run.py         \
  --data-path data.bin \
  --test-id net        \
  --threads 6          \
  --lr 0.001           \
  --wdl 0.5            \
  --max-epochs 65      \
  --batch-size 16384   \
  --save-rate 10       \
  --skip-prop 0.0      \
  --lr-drop 30         \
  --lr-gamma 0.1
```
