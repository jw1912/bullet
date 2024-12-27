use std::{path::Path, process::Command};

use super::util;

pub fn build(out_path: &Path) {
    link_hip_libs(out_path);
    build_and_link_hip_kernels(out_path);
}

fn link_hip_libs(out_path: &Path) {
    let hip_path = util::get_var_path("HIP_PATH");
    let include_path = hip_path.join("include");
    let include_path_str = include_path.to_str().unwrap();

    println!("cargo:rustc-link-arg=-lstdc++");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-search=native={}", hip_path.join("lib").to_str().unwrap());
    println!("cargo:rerun-if-changed={}", include_path_str);
}

fn build_and_link_hip_kernels(out_path: &Path) {
    let files = util::KERNEL_FILES.iter().map(|s| format!("{}/{s}.cu", util::KERNEL_DIR)).collect::<Vec<_>>();

    #[cfg(target_family = "windows")]
    let compiler_name = "hipcc.bin.exe";

    #[cfg(not(target_family = "windows"))]
    let compiler_name = "hipcc";

    // Get the gcnArchName from hipInfo.exe, since hipcc lies about doing it itself
    let gcn_arch_name = get_gcn_arch_name().expect("Failed to get gcnArchName from hipInfo.exe");

    cc::Build::new()
        .compiler(compiler_name)
        .debug(false)
        .opt_level(3)
        .flag(format!("--offload-arch={gcn_arch_name}"))
        .flag("-munsafe-fp-atomics")
        .define("__HIP_PLATFORM_AMD__", None)
        .files(&files)
        .compile("libkernels.a");
}

fn get_gcn_arch_name() -> Option<String> {
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
