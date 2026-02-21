use std::{env, path::PathBuf};

fn get_var_path(name: &str) -> PathBuf {
    println!("rerun-if-env-changed={name}");

    let path = env::var(name).unwrap_or_else(|e| match e {
        env::VarError::NotPresent => panic!("{name} is not defined"),
        env::VarError::NotUnicode(_) => panic!("{name} contains non-unicode path!"),
    });

    println!("Path {name}={path:?}");

    let path = PathBuf::from(path);
    if !path.exists() {
        panic!("Path {name}={path:?} does not exist");
    }

    path
}

fn main() {
    if cfg!(feature = "cuda") {
        let cuda_path = get_var_path("CUDA_PATH");

        let paths = if cfg!(target_family = "windows") { vec!["lib/x64", "lib"] } else { vec!["lib64"] };
        for path in paths {
            println!("cargo:rustc-link-search=native={}", cuda_path.join(path).to_str().unwrap());
        }

        println!("cargo:rerun-if-changed={}", cuda_path.join("include").to_str().unwrap());
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=nvrtc");
        println!("cargo:rustc-link-lib=dylib=cublas");
    }

    if cfg!(feature = "rocm") {
        let hip_path = get_var_path("HIP_PATH");

        println!("cargo:rustc-link-search=native={}", hip_path.join("lib").to_str().unwrap());

        println!("cargo:rerun-if-changed={}", hip_path.join("include").to_str().unwrap());
        println!("cargo:rustc-link-arg=-lstdc++");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=hipblas");
        println!("cargo:rustc-link-lib=dylib=rocblas");
    }
}
