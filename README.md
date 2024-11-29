<div align="center">

# bullet

</div>

bullet is a domain-specific ML library, generally used for training NNUE-style networks for some of the strongest chess engines in the world.

### Features
- Autograd
- CUDA and HIP backends
    - Makes heavy use of (cu/hip)BLAS wherever possible
    - A number of custom kernels
- Lots of NNUE and chess engine specific tooling
    - Input feature types
    - Output buckets
    - Data formats
    - Utilities

### Usage

Before attempting to use, check out the [docs](docs/0-contents.md).
They contain all the main information about building bullet, managing training data and the network output format.

Most people simply clone the repo and edit one of the [examples](/examples) to their taste.

Alternatively, import the crate with
```toml
bullet = { git = "https://github.com/jw1912/bullet" }
```

Specific API documentation is covered by Rust's docstrings.

### Help/Feedback

Please open an issue to file any bug reports/feature requests.
For general questions about bullet - e.g. what your training schedule should look like - you can go to the dedicated channel in the [Engine Programming](https://discord.com/invite/F6W6mMsTGN) discord server.
