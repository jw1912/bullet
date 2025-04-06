use std::{
    env,
    fs::{read_dir, File},
    io::{Read, Write},
    path::PathBuf,
};

use cudarc::nvrtc;

fn main() {
    println!("cargo:rerun-if-changed=./kernels");

    let mut src = String::new();

    let mut file = File::open("./kernels/util.cu").unwrap();
    file.read_to_string(&mut src).unwrap();

    for src_file in read_dir("./kernels").unwrap().filter_map(Result::ok) {
        let name = src_file.file_name();
        let name = name.to_str().unwrap();

        if name == "util.cu" {
            continue;
        }

        let path = format!("./kernels/{name}");
        file = File::open(path).unwrap();
        file.read_to_string(&mut src).unwrap();
    }

    let ptx = nvrtc::compile_ptx(src).unwrap();

    let mut out_path: PathBuf = env::var("OUT_DIR").unwrap().into();
    out_path.push("kernels.ptx");
    let mut out_file = File::create(out_path).unwrap();
    write!(&mut out_file, "{}", ptx.to_src()).unwrap();
}
