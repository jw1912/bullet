<div align="center">

# bullet

</div>

At this point, bullet is a general-purpose neural network trainer,
However, it is generally used for training NNUE-style networks for some of the strongest chess engines in the world.

### Features
- Autograd, the code for which can be found in my [diffable](https://github.com/jw1912/diffable) crate
- CUDA and HIP backends
- Lots of chess specific tooling

### Usage

Before attempting to use, check out the [docs](docs/0-contents.md).
They contain all the main information about building bullet, managing training data and the network output format.

Most people simply clone the repo and edit one of the [examples](/examples) to their taste.

Alternatively, import the crate with
```toml
bullet = { git = "https://github.com/jw1912/bullet" }
```

Specific API documentation is covered by Rust's docstrings.
