use std::path::PathBuf;

pub const KERNEL_DIR: &str = "./src/backend/kernels";

pub const KERNEL_FILES: [&str; 4] = ["activate", "adamw", "add", "power_error"];

pub fn get_var_path(name: &str) -> PathBuf {
    println!("rerun-if-env-changed={}", name);

    use std::env::VarError;
    let path = std::env::var(name).unwrap_or_else(|e| match e {
        VarError::NotPresent => panic!("{name} is not defined"),
        VarError::NotUnicode(_) => panic!("{name} contains non-unicode path!"),
    });

    println!("Path {}={:?}", name, path);

    let path = PathBuf::from(path);
    if !path.exists() {
        panic!("Path {}={:?} does not exist", name, path);
    }

    path
}
