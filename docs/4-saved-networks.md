# Saved Networks

Primitives (e.g. `f32`, `i16`) are always written to files in **little-endian** layout (as is the standard on pretty much all modern hardware).

## Checkpoint Layout

When a checkpoint is saved to a directory `<out_dir>/<checkpoint_name>`, it will contain
- `raw.bin`, the raw floating point (`f32`) parameters of the network
- `quantised.bin`, the quantised network, padded to be a multiple of 64 bytes
- `optimiser_state/`, the internal state of the optimiser

If quantisation fails (due to integer overflow), then it will not save the quantised network, but training will be otherwise unaffected.

## Loading Checkpoints

You can load a preexisting checkpoint into a `trainer: Trainer` by using `trainer.load_from_checkpoint()`.
You can load just the weights from a checkpoint using `trainer.load_weights_from_file(<checkpoint_path>/optimiser_state/weights.bin)`.

## Network Layout with `TrainerBuilder`

If you are using the `TrainerBuilder`, the format of the two network files is `(layer 1 weights)(layer 1 biases)(layer 2 weights)...` stored
contiguously.
The layout in which the weights and biases will be written to file is displayed in the training preamble when you contruct you trainer using `TrainerBuilder`.
