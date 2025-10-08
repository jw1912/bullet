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
is written as `[1, 4, 2, 5, 3, 6]`.

For an affine layer

```rust
let affine = builder.new_affine("affine", input_size, output_size);
```

you can note that the weights are of shape `output_size x input_size`, and the biases are of shape `output_size x 1`.

To save this layer contiguously, quantised to `i16` values with factor `256`, you would add the following `SavedFormat` entries:

```rust
SavedFormat::id("affinew").quantise::<i16>(256),
SavedFormat::id("affineb").quantise::<i16>(256),
```

Suppose you did not want to save the weights column-major, but instead row-major (e.g. for inference performance reasons),
then you would need to transpose the weights:

```rust
SavedFormat::id("affinew").transpose().quantise::<i16>(256)
```

The default behaviour of `.quantise::<T>(Q)` is `quantised_value = truncate(float_value * Q)`.
This is often not desirable, so you can instead change the behaviour to `quantised_value = round(float_value * Q)`
by adding `.round()` like so:

```rust
SavedFormat::id("affinew").round().quantise::<i16>(256),
```

You can apply arbitrary transformations to the float values by chaining `SavedFormat::transform`.
An example of this can be found in the [input buckets example](https://github.com/jw1912/bullet/blob/update-docs/examples/progression/4_multi_layer.rs#L47)
to merge the input factoriser.

The placement of `.round` and `.quantise::<T>` **does not matter**, they are always applied at the end, directly before writing to a file.
All transformations are applied in the order they are specified (and note that `.transpose` simply uses `.transform` internally).
