fn main() {
    #[cfg(feature = "cuda")]
    cuda::build();

    #[cfg(feature = "hip")]
    hip::build();
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::fmt::Debug;
    use std::path::PathBuf;

    use bindgen::callbacks::{MacroParsingBehavior, ParseCallbacks};
    use bindgen::{Builder, CargoCallbacks, EnumVariation};

    const WRAPPER_PATH: &str = "./src/backend/kernels/wrapper.h";

    pub fn build() {
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
            .default_enum_style(EnumVariation::Rust { non_exhaustive: true })
            .must_use_type("cudaError")
            .must_use_type("CUresult")
            .must_use_type("cudaError_enum")
            .layout_tests(false)
            .generate()
            .expect("Unable to generate bindings")
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");

        println!("cargo:rerun-if-changed=./src/backend/kernels");

        let files: Vec<String> =
            ["backprops", "bufops", "mpe", "pairwise_mul", "select", "sparse_affine", "splat_add", "update"]
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

    const IGNORED_MACROS: &[&str] =
        &["FP_INFINITE", "FP_NAN", "FP_NORMAL", "FP_SUBNORMAL", "FP_ZERO", "IPPORT_RESERVED"];

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
}

#[cfg(feature = "hip")]
mod hip {
    use bindgen::callbacks::{MacroParsingBehavior, ParseCallbacks};
    use bindgen::{Builder, CargoCallbacks, EnumVariation};
    use std::env;
    use std::fmt::Debug;
    use std::path::PathBuf;
    use std::process::Command;

    const WRAPPER_PATH: &str = "./src/backend/kernels/hip/wrapper.h";

    pub fn build() {
        let out_path = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let builder = Builder::default();

        // Specify the libraries to link against
        println!("cargo:rustc-link-lib=dylib=hipblas");
        println!("cargo:rustc-link-lib=dylib=rocblas");
        println!("cargo:rustc-link-lib=dylib=amdhip64");

        // Get HIP_PATH and set the library search path
        let hip_path = get_var_path("HIP_PATH");
        let hip_lib_path = hip_path.join("lib");

        // Update the library search path for the linker
        println!("cargo:rustc-link-search=native={}", hip_lib_path.display());

        let include_paths = link_hip();
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
            .default_enum_style(EnumVariation::Rust { non_exhaustive: true })
            .must_use_type("hipError")
            .layout_tests(false)
            .generate()
            .expect("Unable to generate bindings")
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");

        println!("cargo:rerun-if-changed=./src/backend/kernels/hip");

        // Get the gcnArchName from hipInfo.exe, since hipcc lies about doing it itself
        let gcn_arch_name = get_gcn_arch_name().expect("Failed to get gcnArchName from hipInfo.exe");

        let files: Vec<String> =
            ["backprops", "bufops", "mpe", "pairwise_mul", "select", "sparse_affine", "splat_add", "update"]
                .iter()
                .map(|s| format!("./src/backend/kernels/hip/{s}.hip"))
                .collect();

        #[cfg(target_family = "windows")]
        let compiler_name = "hipcc.bin.exe";

        #[cfg(not(target_family = "windows"))]
        let compiler_name = "hipcc.bin";

        cc::Build::new()
            .compiler(compiler_name)
            .debug(false)
            .opt_level(3)
            .files(files)
            .flag(&format!("--offload-arch={}", gcn_arch_name))
            .flag("-munsafe-fp-atomics") // Required since AMDGPU doesn't emit hardware atomics by default
            .compile("libkernels.a");
    }

    fn get_gcn_arch_name() -> Option<String> {
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

    fn link_hip() -> Vec<PathBuf> {
        let path = get_var_path("HIP_PATH");
        println!("cargo:rustc-link-search=native={}", path.join("lib").to_str().unwrap());
        vec![path.join("include")]
    }

    const IGNORED_MACROS: &[&str] =
        &["FP_INFINITE", "FP_NAN", "FP_NORMAL", "FP_SUBNORMAL", "FP_ZERO", "IPPORT_RESERVED"];

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
}
