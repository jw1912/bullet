use std::env;
use std::path::PathBuf;

fn main() {
    hip::build();
}

mod hip {
    use bindgen::{Builder, EnumVariation};
    use std::env;
    use std::path::PathBuf;

    const WRAPPER_PATH: &str = "./src/backend/hip_kernels/wrapper.h";

    pub fn build() {
        let out_path = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        println!("OUT_DIR: {:?}", out_path);
        let builder = Builder::default();

        println!("cargo:rustc-link-lib=dylib=hipblas");
        println!("cargo:rustc-link-lib=dylib=rocblas");
        println!("cargo:rustc-link-lib=dylib=amdhip64");

        let include_paths = link_hip();
        let builder = include_paths.iter().fold(builder, |builder, path| {
            let path = path.to_str().unwrap();
            println!("cargo:rerun-if-changed={}", path);
            builder.clang_arg(format!("-I{}", path))
        });

        println!("cargo:rerun-if-changed={WRAPPER_PATH}");

        let bindings = builder
            .header(WRAPPER_PATH)
            .size_t_is_usize(true)
            .default_enum_style(EnumVariation::Rust { non_exhaustive: true })
            .must_use_type("hipError_t")
            .must_use_type("hipblasStatus_t")
            .layout_tests(false)
            .generate()
            .expect("Unable to generate bindings");

        let bindings_path = out_path.join("bindings.rs");
        println!("Bindings will be written to: {:?}", bindings_path);

        bindings
            .write_to_file(&bindings_path)
            .expect("Couldn't write bindings!");

        println!("cargo:rerun-if-changed=./src/backend/hip_kernels");

        let files: Vec<String> = ["backprops", "bufops", "mpe", "select", "sparse_affine", "splat_add", "update"]
            .iter()
            .map(|s| format!("./src/backend/hip_kernels/{s}.hip"))
            .collect();

        let mut build = cc::Build::new();

        let hip_path = get_var_path("HIP_PATH");
        let hipcc_path = hip_path.join("bin").join("hipcc.bin.exe");
        build.compiler(hipcc_path)
            .cpp(true)
            .shared_flag(true)
            .debug(false)
            .opt_level(3)
            .files(files);

        for path in include_paths {
            build.include(path);
        }

        build.flag("--offload-arch=gfx1100");
        build.compile("libkernels.a");
    }

    fn get_var_path(name: &str) -> PathBuf {
        println!("rerun-if-env-changed={}", name);

        let path = env::var(name).unwrap_or_else(|e| match e {
            env::VarError::NotPresent => panic!("Env Var {name} is not defined"),
            env::VarError::NotUnicode(_) => panic!("Env Var {name} contains non-unicode path!"),
        });

        println!("Path {}={:?}", name, path);

        let path = PathBuf::from(path);
        if !path.exists() {
            panic!("Path {}={:?} does not exist", name, path);
        }

        path
    }

    #[cfg(target_family = "windows")]
    fn link_hip() -> Vec<PathBuf> {
        let path = get_var_path("HIP_PATH");
        println!("cargo:rustc-link-search=native={}", path.join("lib/x64").to_str().unwrap());
        println!("cargo:rustc-link-search=native={}", path.join("lib").to_str().unwrap());
        vec![path.join("include")]
    }

    #[cfg(target_family = "unix")]
    fn link_hip() -> Vec<PathBuf> {
        let path = get_var_path("HIP_PATH");
        println!("cargo:rustc-link-search=native={}", path.join("lib64").to_str().unwrap());
        vec![path.join("include")]
    }
}
