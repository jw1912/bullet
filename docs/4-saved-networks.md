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

## Network Layout with `TrainerBuilder`

If you are using the `TrainerBuilder`, the format of the two network files is `(layer 1 weights)(layer 1 biases)(layer 2 weights)...` stored
contiguously, with the weights matrices being stored column-major.

### Column Major

Column major means that the following matrix

$$
\begin{bmatrix}
    x_{11} & x_{12} & x_{13} & x_{14} \\
    x_{21} & x_{22} & x_{23} & x_{24} \\
    x_{31} & x_{32} & x_{33} & x_{34}
\end{bmatrix}
$$

is stored as

$$
    x_{11}, x_{21}, x_{31},
    x_{12}, x_{22}, x_{32},
    x_{13}, x_{23}, x_{33},
    x_{14}, x_{24}, x_{34}
$$

in memory.

### Output Buckets Weights

Now, if you have output buckets `A`, `B`, `C` for an affine layer `4 -> 3`, the weights matrix representing them is 

$$
\begin{bmatrix}
    A_{11} & A_{12} & A_{13} & A_{14} \\
    A_{21} & A_{22} & A_{23} & A_{24} \\
    A_{31} & A_{32} & A_{33} & A_{34} \\
    B_{11} & B_{12} & B_{13} & B_{14} \\
    B_{21} & B_{22} & B_{23} & B_{24} \\
    B_{31} & B_{32} & B_{33} & B_{34} \\
    C_{11} & C_{12} & C_{13} & C_{14} \\
    C_{21} & C_{22} & C_{23} & C_{24} \\
    C_{31} & C_{32} & C_{33} & C_{34}
\end{bmatrix}
$$

and **in particular**, for a layer `4 -> 1`, we have

$$
\begin{bmatrix}
    A_{1} & A_{2} & A_{3} & A_{4} \\
    B_{1} & B_{2} & B_{3} & B_{4} \\
    C_{1} & C_{2} & C_{3} & C_{4} 
\end{bmatrix}
$$

so in memory, the weights for each bucket are stored as

$$
    A_1, B_1, C_1, A_2, B_2, C_2, A_3, B_3, C_3, A_4, B_4, C_4
$$

this is important as for fast inference on the CPU, you will want to **transpose** these weights to allow weights from the same bucket to be contiguous in memory.

### Output Buckets Biases

These are stored contiguously per bucket, so in an affine layer with output size 3
and buckets `A`, `B`, `C`, these are stored as

$$
A_1, A_2, A_3, B_1, B_2, B_3, C_1, C_2, C_3
$$

in memory.
