mod cuda;
mod util;

use std::path::PathBuf;

fn main() {
    let out_path = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=./src/backend/kernels");

    cuda::build(out_path);
}
