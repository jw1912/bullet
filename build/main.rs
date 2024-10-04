#[allow(unused)]
mod cuda;
#[allow(unused)]
mod hip;
mod util;

#[cfg(not(feature = "gh-actions"))]
fn main() {
    let out_path = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed={}", util::KERNEL_DIR);

    #[cfg(not(feature = "hip"))]
    cuda::build(&out_path);

    #[cfg(feature = "hip")]
    hip::build(&out_path);
}

#[cfg(feature = "gh-actions")]
fn main() {}
