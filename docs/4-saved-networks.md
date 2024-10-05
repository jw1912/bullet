## Saved Networks

When a checkpoint is saved to a directory `<out_dir>/<checkpoint_name>`, it will contain
- `raw.bin`, the raw floating point (`f32`) parameters of the network
- `quantised.bin`, the quantised network, padded to be a multiple of 64 bytes
- `optimiser_state/`, the internal state of the optimiser

If quantisation fails (due to integer overflow), then it will not save the quantised network, but training will be otherwise unaffected.

If you are using the `TrainerBuilder`, the format of the two network files is `(layer 1 weights)(layer 1 biases)(layer 2 weights)...` stored
contiguously, with the weights matrices being stored column-major.
