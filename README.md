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

- **bullet_core**
    - An ML framework that is generic over backends
    - Graphs are defined once (ahead of use), then optimsed and compiled for a given backend device
    - A token single-threaded CPU backend is included for verifying correctness of the crate and other backend implementations
    - See the [MNIST](examples/extra/mnist.rs) example for using `bullet_core` as a general-purpose ML framework
- **bullet_cuda_backend**
    - A working but incomplete CUDA backend rewrite, not currently suitable for serious use
- **bullet_hip_backend**
    - Currently contains both the HIP (for AMD GPUs) and CUDA backends. Enable the `hip` feature to use the HIP backend
- **bullet_lib**
    - Provides a high-level wrapper around the above crates specifically for training networks to do with chess (and other games e.g. Ataxx) easily
    - Value network training for games with `ValueTrainer`
        - The [simple](examples/simple.rs) example shows ease-of-use in training the simplest NNUE architectures
        - The [progression](examples/progression) examples show how to incrementally improve your NNUE architecture
    - What backend is used is dictated by passed feature flags:
        - By default the CUDA backend from `bullet_hip_backend` is used, you should not pass any feature flags if you want to use the CUDA backend
        - Enable the `hip` feature to use the HIP backend **only** if you have an AMD card
        - Read the [documentation](docs/2-getting-started.md#backends) for more specific instructions
- **bullet-utils**
    - Various utilities mostly to do with handling chess data

### Help/Feedback

- Please open an issue to file any bug reports/feature requests.
- Feel free to use the dedicated `#bullet` channel in the [Engine Programming](https://discord.com/invite/F6W6mMsTGN) discord server if you run into any issues.
- For general training discussion the Engine Programming non-`#bullet` channels are appropriate, or `#engines-dev` in the [Stockfish](https://discord.gg/GWDRS3kU6R) discord.
