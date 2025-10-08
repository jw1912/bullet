# Saved Networks

## Checkpoint Layout

When a checkpoint is saved to a directory `<out_dir>/<checkpoint_name>`, it will contain
- `raw.bin`, the raw floating point (`f32`) parameters of the network
- `quantised.bin`, the quantised network, padded to be a multiple of 64 bytes
- `optimiser_state/`, the internal state of the optimiser

If quantisation fails (due to integer overflow), then it will not save the quantised network, but training will be otherwise unaffected.

## Loading Checkpoints

You can load a preexisting checkpoint into a `trainer: Trainer` by using `trainer.load_from_checkpoint()`.
You can load just the weights from a checkpoint using `trainer.load_weights_from_file(<checkpoint_path>/optimiser_state/weights.bin)`.

## Layout of `SavedFormat`

Primitives (e.g. `f32`, `i16`) are always written to files in **little-endian** layout (as is the standard on pretty much all modern hardware).

Every weight has an associated shape, `MxN`, and is written in **column-major** format.

This means the following 2x3 matrix:

```
[1, 2, 3]
[4, 5, 6]
```
is written as [1, 4, 2, 5, 3, 6].

For an affine layer

```rust
let affine = builder.new_affine("affine", input_size, output_size);
```

you can note that the weights are of shape `output_size x input_size`, and the biases are of shape `output_size x 1`.
