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

At present all trainer settings are constants at the top of `trainer/src/main.rs`, although this will change soon.

To run the trainer, change to your preferred settings and run
```
cargo r -r --bin trainer <data file path>
```
