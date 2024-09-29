use std::path::{Path, PathBuf};

use bindgen::{Builder, EnumVariation};

use super::util;

pub fn build(out_path: PathBuf) {
    link_cuda_libs(out_path);

    build_and_link_cuda_kernels();
}

fn link_cuda_libs(out_path: PathBuf) {
    let builder = Builder::default();
    let cuda_path = util::get_var_path("CUDA_PATH");
    let include_path = cuda_path.join("include");
    let include_path_str = include_path.to_str().unwrap();

    println!("cargo:rustc-link-lib=static=cublas");
    link_cuda(&cuda_path);
    println!("cargo:rerun-if-changed={}", include_path_str);

    builder
        .clang_arg(format!("-I{}", include_path_str))
        .header_contents("wrapper.h", "#include <cublas_v2.h>")
        .size_t_is_usize(true)
        .default_enum_style(EnumVariation::Rust { non_exhaustive: true })
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn build_and_link_cuda_kernels() {
    let files: Vec<String> =
        ["backprops", "bufops", "mpe", "pairwise_mul", "select", "sparse", "splat_add", "update"]
            .iter()
            .map(|s| format!("./src/backend/kernels/{s}.cu"))
            .collect();

    cc::Build::new()
        .cuda(true)
        .cudart("shared")
        .debug(false)
        .opt_level(3)
        .include("cuda")
        .include("")
        .files(files)
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