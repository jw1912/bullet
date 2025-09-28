<div align="center">

# bullet

</div>

A domain-specific ML library, generally used for training NNUE-style networks for many of the strongest chess engines in the world
due to its best-in-class performance, chess-specific tooling and ease of use.

### Usage for NNUE/Value Network Training

Before attempting to use, check out the [docs](docs/0-contents.md).
They contain all the main information about building bullet, managing training data and the network output format.

Most people simply clone the repo and edit one of the [examples](/examples) to their taste.
If you want to create your own example file to ease pulling from upstream, you need to add the example to [`bullet_lib`'s `Cargo.toml`](crates/bullet_lib/Cargo.toml).

Alternatively, import the `bullet_lib` crate with
```toml
bullet = { git = "https://github.com/jw1912/bullet", package = "bullet_lib" }
```

Specific API documentation is covered by Rust's docstrings. You can create local documentations with `cargo doc`.

### Constituent Crates

- **acyclib**
    - Core ML library for directed acyclic tensor graphs
    - Graphs are defined once (ahead of use), then optimised and compiled for a given backend device
    - Contains the CPU backend
    - Contains training abstractions to take care of asynchronous data loading and transfer from CPU to GPU
    - See the [MNIST](examples/extra/mnist.rs) example for using it at a lower level
- **bullet_cuda_backend**
    - The first-class backend
    - Things that other backends don't have:
        - Tooling to make writing custom operations easier
        - Additional optimisation passes & better optimised kernels
        - Multi-GPU support! See the [caveat](https://github.com/jw1912/bullet/commit/b06dd9bbbcfde9716612f0d305d1d94279a26a04) for whether bandwidth limitations will be a performance bottleneck
- **bullet_hip_backend**
    - For users with AMD GPUs
- **bullet_lib**
    - Provides a high-level wrapper around the above crates specifically for training networks to do with chess (and other games e.g. Ataxx) easily
    - Value network training for games with `ValueTrainer`
        - The [simple](examples/simple.rs) example shows ease-of-use in training the simplest NNUE architectures
        - The [progression](examples/progression) examples show how to incrementally improve your NNUE architecture
        - Read the [documentation](docs/2-getting-started.md#backends) for compilation instructions
- **bullet-utils**
    - Various utilities mostly to do with handling chess data

### Help/Feedback

- Please open an issue to file any bug reports/feature requests.
- Feel free to use the dedicated `#bullet` channel in the [Engine Programming](https://discord.com/invite/F6W6mMsTGN) discord server if you run into any issues.
- For general training discussion the Engine Programming non-`#bullet` channels are appropriate, or `#engines-dev` in the [Stockfish](https://discord.gg/GWDRS3kU6R) discord. 
