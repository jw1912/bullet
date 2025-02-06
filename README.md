<div align="center">

# bullet

</div>

bullet is a domain-specific ML library, generally used for training NNUE-style networks for some of the strongest chess engines in the world.

### Crates & Versioning

#### [bullet_core](crates/bullet_core)

- Follows SemVer with crates.io releases
- Contains the `Device` and `DeviceBuffer` traits, which are used to define a backend
- Network graph construction, execution and autodiff generic over backends

#### [bullet_cuda_backend](crates/bullet_cuda_backend)

- Follows SemVer with crates.io releases
- New CUDA backend
- Not currently in a useable state

#### [bullet_hip_backend](crates/bullet_hip_backend)

- Follows SemVer with crates.io releases
- Contains both the HIP and CUDA backends

#### [bullet_lib](crates/bullet_lib)

- Does not follow any particular versioning
- API is sometimes subject to breaking changes
- `NetworkTrainer` traits wrap the core graph API, providing a training loop with data loading performed asynchronously from device calculations
- Contains `Trainer`, which implements `NetworkTrainer` for value network training
- Contains `TrainerBuilder`, which streamlines the process of constructing a `Trainer` for the most common network architectures
- Lots of NNUE and chess engine specific tooling
    - Input feature types
    - Output buckets
    - Data formats

#### [bullet-utils](crates/bullet_utils)

- Does not follow any particular versioning
- Is a CLI program with various chess related utilities
- Data validation
- Converting between data file types
- Shuffling and interleaving data files

### Usage for Value Network Training

Before attempting to use, check out the [docs](docs/0-contents.md).
They contain all the main information about building bullet, managing training data and the network output format.

Most people simply clone the repo and edit one of the [examples](/examples) to their taste.

Alternatively, import the `bullet_lib` crate with
```toml
bullet = { git = "https://github.com/jw1912/bullet", package = "bullet_lib" }
```

Specific API documentation is covered by Rust's docstrings.

### Help/Feedback

- Please open an issue to file any bug reports/feature requests.
- Feel free to use the dedicated `#bullet` channel in the [Engine Programming](https://discord.com/invite/F6W6mMsTGN) discord server if you run into any issues.
- For general training discussion the Engine Programming non-`#bullet` channels are appropriate, or `#engines-dev` in the [Stockfish](https://discord.gg/GWDRS3kU6R) discord.
