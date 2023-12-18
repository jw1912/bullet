<div align="center">

# bullet

</div>

A Neural Network Trainer, used to train NNUE-style networks for [akimbo](https://github.com/jw1912/akimbo).

Also used by a number of other engines, including:
- [Alexandria](https://github.com/PGG106/Alexandria)
- [Altair](https://github.com/Alex2262/AltairChessEngine)
- [Carp](https://github.com/dede1751/carp)
- [Midnight](https://github.com/archishou/MidnightChessEngine)
- [Obsidian](https://github.com/gab8192/Obsidian)
- [Stormphrax](https://github.com/Ciekce/Stormphrax)
- [Willow](https://github.com/Adam-Kulju/Willow)

## About

Used exclusively to train architectures of the form `Input -> Nx2 -> Output`.

Can use either a CPU or hand-written CUDA backend.

Currently Supported Games:
- Chess
- Ataxx

Raise an issue for support of a new game.

To learn how it works, read the [wiki](wiki.md).

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
- `FEN` has 'r', 'b' and '-' for red, blue and gaps/blockers, respectively, in the same format as FEN for chess
- `score` is red relative and an integer
- `result` is red relative and of the form `1.0` for win, `0.5` for draw, `0.0` for loss

### Chess

You can convert a [Marlinformat](https://github.com/jnlt3/marlinflow) file by running
```
cargo r -r --bin convertmf <input file path> <output file path>
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

## Training

General architecture settings, that must be known at compile time, are found in [`common/src/lib.rs`](common/src/lib.rs).
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
  --max-epochs 40      \
  --batch-size 16384   \
  --save-rate 10       \
  --lr-step 15         \
  --lr-gamma 0.1
```

of these options, only `data-path`, `threads` and `lr-step` are not default values.

NOTE: You may need to run `cargo update` if you pull a newer version from `main`.

### Learning Rate Scheduler
There are 3 separate learning rate options:
- `lr-step N` drops the learning rate every `N` epochs by a factor of `lr-gamma`
- `lr-drop N` drops the learning rate once, at `N` epochs, by a factor of `lr-gamma`
- `lr-end x` is exponential LR, starting at `lr` and ending at `x` when at `max-epochs`,
it is equivalent to `lr-step 1` with an appropriate `lr-gamma`.

By default `lr-gamma` is set to 0.1, but no learning rate scheduler is chosen. It is highly
recommended to have at least one learning rate drop during training.

### CUDA

Add `--cuda` to use CUDA, it will fail to compile if not available.
It is not recommended to use CUDA for small net sizes (unbucketed & hidden layer < 256).

### AVX512

Currently (at the time of writing) rustc does not emit avx512 via autovec, so if you have an avx512 cpu, switch to the nightly
Rust channel and add the `--simd` flag to the run command to enable usage of hand-written SIMD.
This comes with the caveat that hidden layer size must be a multiple of 32.

As rust nightly is unstable
and has a bunch of experimental compiler changes, there may be an overall diminished performance compared
to compiling on stable, so I'd recommend testing the two on your machine.

### Resuming

Every `save-rate` epochs and at the end of training, a quantised network is saved to `/nets`, and a checkpoint
is saved to `/checkpoints` (which contains the raw network params, if you want them). You can "resume" from a checkpoint by
adding `--resume checkpoints/<name of checkpoint folder>` to the run command.

This is designed such that if you use an identical
command with the resuming appended, it would be as if you never stopped training, so if using a different command be wary that
it will try to resume from the epoch number the checkpoint was saved at, meaning it will fast-forward Learning Rate to that epoch.
