#[allow(unused)]
mod cuda;
#[allow(unused)]
mod hip;
mod util;

use std::path::PathBuf;

fn main() {
    let out_path = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed={}", util::KERNEL_DIR);

    #[cfg(not(feature = "hip"))]
    cuda::build(&out_path);

    #[cfg(feature = "hip")]
    hip::build(&out_path);
}
