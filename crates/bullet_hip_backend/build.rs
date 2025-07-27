#![allow(unused)]
use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    if !cfg!(feature = "gh-actions") {
        let out_path = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

        println!("cargo:rerun-if-changed=./kernels");

        if cfg!(feature = "hip") {
            build_hip(&out_path);
        } else {
            build_cuda(&out_path);
        }
    }
}

const KERNELS: &str = "./kernels/include.cu";

fn get_var_path(name: &str) -> PathBuf {
    println!("rerun-if-env-changed={name}");

    use std::env::VarError;
    let path = std::env::var(name).unwrap_or_else(|e| match e {
        VarError::NotPresent => panic!("{name} is not defined"),
        VarError::NotUnicode(_) => panic!("{name} contains non-unicode path!"),
    });

    println!("Path {name}={path:?}");

    let path = PathBuf::from(path);
    if !path.exists() {
        panic!("Path {name}={path:?} does not exist");
    }

    path
}

fn link_search(base_path: &Path) {
    let paths = if cfg!(target_family = "windows") { vec!["lib/x64", "lib"] } else { vec!["lib64"] };

    for path in paths {
        println!("cargo:rustc-link-search=native={}", base_path.join(path).to_str().unwrap());
    }
}

fn build_cuda(out_path: &Path) {
    let cuda_path = get_var_path("CUDA_PATH");
    let include_path = cuda_path.join("include");
    let include_path_str = include_path.to_str().unwrap();

    println!("cargo:rustc-link-lib=dylib=cublas");
    link_search(&cuda_path);
    println!("cargo:rerun-if-changed={include_path_str}");

    cc::Build::new()
        .cargo_warnings(false)
        .cuda(true)
        .cudart("shared")
        .debug(false)
        .flag("-arch=native")
        .opt_level(3)
        .files(&[KERNELS])
        .out_dir(out_path)
        .compile("libkernels.a");
}

fn build_hip(out_path: &Path) {
    let hip_path = get_var_path("HIP_PATH");
    let include_path = hip_path.join("include");
    let include_path_str = include_path.to_str().unwrap();
    let compiler_name = if cfg!(target_family = "windows") { "hipcc.bin.exe" } else { "hipcc" };
    let gcn_arch_name = get_gcn_arch_name().expect("Failed to get gcnArchName from hipInfo.exe");

    println!("cargo:rustc-link-arg=-lstdc++");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-search=native={}", hip_path.join("lib").to_str().unwrap());
    println!("cargo:rerun-if-changed={include_path_str}");

    cc::Build::new()
        .cargo_warnings(false)
        .warnings(false)
        .compiler(compiler_name)
        .debug(false)
        .opt_level(3)
        .flag(format!("--offload-arch={gcn_arch_name}"))
        .flag("-munsafe-fp-atomics")
        .define("__HIP_PLATFORM_AMD__", None)
        .out_dir(out_path)
        .files(&[KERNELS])
        .compile("libkernels.a");
}

fn get_gcn_arch_name() -> Option<String> {
    println!("rerun-if-env-changed=GCN_ARCH_NAME");
    let provided = std::env::var("GCN_ARCH_NAME").ok();

    if provided.is_some() {
        return provided;
    }

    let output = Command::new("hipInfo").output().expect("Failed to execute hipInfo.exe");

    if output.status.success() {
        let output_str = String::from_utf8_lossy(&output.stdout);
        for line in output_str.lines() {
            if line.contains("gcnArchName:") {
                return line.split_whitespace().last().map(String::from);
            }
        }
    }

    None
}
