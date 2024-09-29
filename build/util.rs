use std::path::PathBuf;

pub fn get_var_path(name: &str) -> PathBuf {
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