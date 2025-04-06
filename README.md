<div align="center">

# bullet

</div>

A domain-specific ML library, generally used for training NNUE-style networks for some of the strongest chess engines in the world.

### Crates

- **bullet_core**
    - An ML framework that is generic over backends:
    - A network graph is constructed using `GraphBuilder`, which internally generates a `GraphIR`
    - [Optimisation passes](docs/advanced-examples/operator-fusion.md) are performed on the `GraphIR`
    - The `GraphIR` is then compiled into a `Graph<D: Device>`, for a specific backend device
        - Upon which forwards and backwards passes, editing weights/inputs, etc, may be performed
        - A small set of (composable) optimisers are included that ingest a graph and provide update methods for it
    - A token single-threaded CPU backend is included for verifying correctness of the crate and other backend implementations
    - See the [MNIST](examples/extra/mnist.rs) example for using `bullet_core` as a general-purpose ML framework
- **bullet_cuda_backend**
    - A working but incomplete CUDA backend rewrite, not currently suitable for serious use.
- **bullet_hip_backend**
    - Currently contains both the HIP (for AMD GPUs) and CUDA backends. Enable the `hip` feature to use the HIP backend.
- **bullet_lib**
    - Provides a high-level wrapper around the above crates specifically for training networks to do with chess (and other games e.g. Ataxx) easily.
    - Value network training for games with `Trainer`
        - The [simple](examples/simple.rs) example shows ease-of-use in training the simplest NNUE architectures
        - The [advanced](examples/advanced.rs) example shows how to train flexible value network architectures
    - Policy network training for chess with `PolicyTrainer`, see the [policy](examples/extra/policy.rs) example
- **bullet-utils**
    - Various utilities mostly to do with handling data

### Usage for NNUE/Value Network Training

Before attempting to use, check out the [docs](docs/0-contents.md).
They contain all the main information about building bullet, managing training data and the network output format.

Most people simply clone the repo and edit one of the [examples](/examples) to their taste.
If you want to create your own example file to ease pulling from upstream, you need to add the example to [`bullet_lib`'s `Cargo.toml`](crates/bullet_lib/Cargo.toml).

Alternatively, import the `bullet_lib` crate with
```toml
bullet = { git = "https://github.com/jw1912/bullet", package = "bullet_lib" }
```

Specific API documentation is covered by Rust's docstrings.

### Help/Feedback

- Please open an issue to file any bug reports/feature requests.
- Feel free to use the dedicated `#bullet` channel in the [Engine Programming](https://discord.com/invite/F6W6mMsTGN) discord server if you run into any issues.
- For general training discussion the Engine Programming non-`#bullet` channels are appropriate, or `#engines-dev` in the [Stockfish](https://discord.gg/GWDRS3kU6R) discord.
