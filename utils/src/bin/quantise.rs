use cpu::{quantise_and_write, NetworkParams};

use std::env::args;

fn main() {
    let inp_path = args().nth(1).expect("Expected a file name!");
    let out_path = args().nth(2).expect("Expected a file name!");
    let qa = args().nth(3).expect("Expected an integer for `qa`!").parse().unwrap();
    let qb = args().nth(4).expect("Expected an integer for `qb`!").parse().unwrap();

    let mut net = NetworkParams::new();
    net.load_from_bin(inp_path.as_str());

    quantise_and_write(&net, out_path.as_str(), qa, qb)
}