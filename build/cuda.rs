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

    println!("cargo:rustc-link-lib=dylib=cublas");
    link_cuda(&cuda_path);
    println!("cargo:rerun-if-changed={}", include_path_str);

    #[cfg(feature = "cudnn")]
    cudnn::link_cudnn_libs();
}

fn build_and_link_cuda_kernels(out_path: &Path) {
    let files = util::KERNEL_FILES.iter().map(|s| format!("{}/{s}.cu", util::KERNEL_DIR)).collect::<Vec<_>>();

    cc::Build::new()
        .cargo_warnings(false)
        .cuda(true)
        .cudart("shared")
        .debug(false)
        .opt_level(3)
        .files(&files)
        .out_dir(out_path)
        .compile("libkernels.a");
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

mod cudnn {
    use std::path::Path;

    use crate::util;

    pub fn link_cudnn_libs() {
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cudnn");
        println!("cargo:rustc-link-lib=dylib=cublas");
        let cudnn_path = util::get_var_path("CUDNN_PATH");
        let include_path = cudnn_path.join("include");
        let include_path_str = include_path.to_str().unwrap();
        link_cudnn(&cudnn_path);
        println!("cargo:rerun-if-changed={}", include_path_str);
    }

    #[cfg(target_family = "windows")]
    fn link_cudnn(cudnn_path: &Path) {
        println!("cargo:rustc-link-search=native={}", cudnn_path.join("lib/x64").to_str().unwrap());
        println!("cargo:rustc-link-search=native={}", cudnn_path.join("lib").to_str().unwrap());
    }

    #[cfg(target_family = "unix")]
    fn link_cudnn(cudnn_path: &Path) {
        println!("cargo:rustc-link-search=native={}", cudnn_path.join("lib64").to_str().unwrap());
    }
}
