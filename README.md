<div align="center">

# bullet

</div>

A domain-specific ML library, generally used for training NNUE-style networks for many of the strongest chess engines in the world
due to its best-in-class performance, chess-specific tooling and ease of use. Training is supported on NVIDIA (`cuda`), AMD (`rocm`) and Apple Silicon (`metal`) GPUs.

### Usage for NNUE/Value Network Training

Before attempting to use, check out the [docs](docs/0-contents.md) which contain the main information about building bullet, managing training data and the network output format.

Most people simply clone the repo and edit one of the [examples](/examples) to their taste:
- [`simple`](examples/simple.rs) - a basic `(768 -> N)x2 -> 1` network, with example inference code, that uses `ValueTrainerBuilder`
- [`progression`](examples/progression) - from a first network to input buckets, output buckets and multiple layers
- [`advanced`](examples/advanced) - a SOTA training example that does not use `ValueTrainerBuilder`, instead using `bullet-trainer` directly
- [`ataxx`](examples/ataxx.rs) - a non-chess example, where a simple custom data format is defined

If you want to create your own example file to ease pulling from upstream, you need to add the example to [`bullet_lib`'s `Cargo.toml`](crates/bullet_lib/Cargo.toml).

Alternatively, import the `bullet_lib` crate with
```toml
bullet = { git = "https://github.com/jw1912/bullet", package = "bullet_lib" }
```

Specific API documentation is covered by Rust's docstrings. You can create local documentations with `cargo doc`.

### Help/Feedback

Open an issue to file any simple bug reports/feature requests.
The `#bullet` channel in the [Engine Programming](https://discord.com/invite/F6W6mMsTGN) discord server exists for help with the use of bullet and discussing feature requests, development and/or potential bugs.
It is **not** for questions about how you should train your network - for that refer to `#engines-dev` in [Stockfish](https://discord.gg/GWDRS3kU6R) or `#nnue-dev` in [Alpha-Beta](https://discord.gg/t3aX6XkPaV).
