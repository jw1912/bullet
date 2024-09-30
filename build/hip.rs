use std::path::Path;

use super::util;

pub fn build(out_path: &Path) {
    link_hip_libs(out_path);
    //build_and_link_hip_kernels(out_path);
}

fn link_hip_libs(out_path: &Path) {
    let hip_path = util::get_var_path("HIP_PATH");
    let include_path = hip_path.join("include");
    let include_path_str = include_path.to_str().unwrap();

    println!("cargo:rustc-link-lib=static=hipblas");
    println!("cargo:rustc-link-lib=static=rocmblas");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-search=native={}", hip_path.join("lib").to_str().unwrap());
    println!("cargo:rerun-if-changed={}", include_path_str);

    let header = "#define __HIP_PLATFORM_AMD__\n#include <hipblas/hipblas.h>";

    bindgen::Builder::default()
        .clang_arg(format!("-I{}", include_path_str))
        .header_contents("wrapper.h", header)
        .size_t_is_usize(true)
        .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: true })
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn build_and_link_hip_kernels(out_path: &Path) {
    let files = util::KERNEL_FILES
        .iter()
        .map(|s| format!("{}/{s}.cu", util::KERNEL_DIR))
        .collect::<Vec<_>>();

    #[cfg(target_family = "windows")]
    let compiler_name = "hipcc.bin.exe";

    #[cfg(not(target_family = "windows"))]
    let compiler_name = "hipcc";

    cc::Build::new()
        .compiler(compiler_name)
        .flag("-munsafe-fp-atomics")
        .files(&files)
        .out_dir(out_path)
        .compile("libkernels.a");
}
