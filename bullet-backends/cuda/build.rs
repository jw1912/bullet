use std::fmt::Debug;
use std::path::PathBuf;

use bindgen::callbacks::{MacroParsingBehavior, ParseCallbacks};
use bindgen::{Builder, CargoCallbacks, EnumVariation};

const WRAPPER_PATH: &str = "./kernels/wrapper.h";

fn main() {
    let out_path = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let builder = Builder::default();

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cublas");

    let include_paths = link_cuda();
    let builder = include_paths.iter().fold(builder, |builder, path| {
        let path = path.to_str().unwrap();

        println!("cargo:rerun-if-changed={}", path);
        builder.clang_arg(format!("-I{}", path))
    });

    println!("cargo:rerun-if-changed={WRAPPER_PATH}");

    builder
        .header(WRAPPER_PATH)
        .parse_callbacks(Box::new(CustomParseCallBacks))
        .size_t_is_usize(true)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: true,
        })
        .must_use_type("cudaError")
        .must_use_type("CUresult")
        .must_use_type("cudaError_enum")
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("cargo:rerun-if-changed=./kernels");

    let files: Vec<String> = [
        "backprops",
        "bufops",
        "mse",
        "select",
        "sparse_affine",
        "splat_add",
        "update",
    ]
    .iter()
    .map(|s| format!("./kernels/{s}.cu"))
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

fn get_var_path(name: &str) -> PathBuf {
    println!("rerun-if-env-changed={}", name);

    use std::env::VarError;
    let path = std::env::var(name).unwrap_or_else(|e| match e {
        VarError::NotPresent => panic!("Env Var {name} is not defined"),
        VarError::NotUnicode(_) => panic!("Env Var {name} contains non-unicode path!"),
    });

    println!("Path {}={:?}", name, path);

    let path = PathBuf::from(path);
    if !path.exists() {
        panic!("Path {}={:?} does not exist", name, path);
    }

    path
}

#[cfg(target_family = "windows")]
fn link_cuda() -> Vec<PathBuf> {
    let path = get_var_path("CUDA_PATH");
    println!("cargo:rustc-link-search=native={}", path.join("lib/x64").to_str().unwrap());
    println!("cargo:rustc-link-search=native={}", path.join("lib").to_str().unwrap());
    vec![path.join("include")]
}

#[cfg(target_family = "unix")]
fn link_cuda() -> Vec<PathBuf> {
    let path = get_var_path("CUDA_PATH");
    println!("cargo:rustc-link-search=native={}", path.join("lib64").to_str().unwrap());
    vec![path.join("include")]
}

const IGNORED_MACROS: &[&str] = &[
    "FP_INFINITE",
    "FP_NAN",
    "FP_NORMAL",
    "FP_SUBNORMAL",
    "FP_ZERO",
    "IPPORT_RESERVED",
];

#[derive(Debug)]
struct CustomParseCallBacks;

impl ParseCallbacks for CustomParseCallBacks {
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
        if IGNORED_MACROS.contains(&name) {
            MacroParsingBehavior::Ignore
        } else {
            MacroParsingBehavior::Default
        }
    }

    // redirect to normal handler
    fn include_file(&self, filename: &str) {
        CargoCallbacks.include_file(filename)
    }
}
