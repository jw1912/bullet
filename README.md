<div align="center">

# bullet

</div>

A work-in-progress NNUE Trainer, used to train [akimbo](https://github.com/JacquesRW/akimbo)'s networks.

It currently supports architectures of the form `768 -> Nx2 -> 1`, and can train on CPU with any number of threads.

## Usage

### Data

The trainer uses [Marlinformat](https://github.com/jnlt3/marlinflow) as its primary binary data format.

You can convert EPD files of the form `<FEN> <score> <win/draw/loss>` to this format by running
```
cargo r -r --bin convert <input file path> <output file path>
```

Note that FENs *must* be full, including halfmove and fullmove counters (though they can be set to any valid values).

### Training

General architecture settings, that must be known at compile time, are found in [`trainer/src/lib.rs`](trainer/src/lib.rs).

After settings those as you please, you can run the trainer using the `run.py` script, and use
```
python3 run.py --help
```
to get a full description of all options.

A sample usage is
```
python3 run.py         \
  --data-path data.bin \
  --test-id net001     \
  --threads 1          \
  --lr 0.001           \
  --wdl 0.5            \
  --max-epochs 65      \
  --batch-size 16384   \
  --save-rate 10       \
  --skip-prop 0.0
```
