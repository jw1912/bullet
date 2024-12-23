<div align="center">

# bullet-legacy

</div>

This is the legacy branch of bullet, intended for fast CPU training of chess networks with architecture `(INPUTS -> HL_SIZE)x2 -> 1xOUTPUT_BUCKETS`.

All the options for training are found in `src/main.rs`, and you run the trainer with `cargo r -r`.

The docs from the main branch of bullet still apply and you should use `bullet-utils` from the main branch for dealing with data.
