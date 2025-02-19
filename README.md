<div align="center">

# bullet

</div>

A domain-specific ML library, generally used for training NNUE-style networks for some of the strongest chess engines in the world.

### Crates

The main crate is `bullet_core`, which is an ML framework that is generic over backends.
There are then crates for specific backend implementations, e.g. `bullet_hip_backend`.
The crate `bullet_lib` provides a high-level wrapper around these crates specifically for training
networks to do with chess (and other games e.g. Ataxx) easily. 
There are various misc utilities mostly to do with handling data in `bullet-utils`.

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
