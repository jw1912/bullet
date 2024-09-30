use std::path::Path;

use super::util;

pub fn build(out_path: &Path) {
    link_cuda_libs();
    build_and_link_cuda_kernels(out_path);
}

fn link_cuda_libs() {
    let cuda_path = util::get_var_path("CUDA_PATH");
    let include_path = cuda_path.join("include");
    let include_path_str = include_path.to_str().unwrap();

    println!("cargo:rustc-link-lib=static=cublas");
    link_cuda(&cuda_path);
    println!("cargo:rerun-if-changed={}", include_path_str);
}

fn build_and_link_cuda_kernels(out_path: &Path) {
    let files = util::KERNEL_FILES.iter().map(|s| format!("{}/{s}.cu", util::KERNEL_DIR)).collect::<Vec<_>>();

    cc::Build::new().cuda(true).cudart("shared").files(&files).out_dir(out_path).compile("libkernels.a");
}

#[cfg(target_family = "windows")]
fn link_cuda(cuda_path: &Path) {
    println!("cargo:rustc-link-search=native={}", cuda_path.join("lib/x64").to_str().unwrap());
    println!("cargo:rustc-link-search=native={}", cuda_path.join("lib").to_str().unwrap());
}

#[cfg(target_family = "unix")]
fn link_cuda(cuda_path: &Path) {
    println!("cargo:rustc-link-search=native={}", cuda_path.join("lib64").to_str().unwrap());
}
