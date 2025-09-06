fn main() {
    #[cfg(feature = "error")]
    compile_error!(
        "\
        You have enabled the `cpu`, `hip` or `hip-cuda` feature on bullet_lib but have not disabled default features!\
        \nIf you are running an example, try the following:\
        \n    cargo r -r --example <example name> --features hip --no-default-features
        \nIf you are using bullet as a crate, ammend your Cargo.toml to
        \n    bullet_lib = { ... other stuff ..., default-features = false }
    "
    );
}
