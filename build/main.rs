mod cuda;
mod util;

use std::path::PathBuf;

fn main() {
    let out_path = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed={}", util::KERNEL_DIR);

    cuda::build(&out_path);
}
