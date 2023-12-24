use std::env::args;

use bulletformat::{ChessBoard, chess::MarlinFormat, convert_from_bin};

fn main() {
    let inp_path = args().nth(1).expect("Expected a file name!");
    let out_path = args().nth(2).expect("Expected a file name!");
    let threads = args().nth(3).expect("Expected number of threads!").parse().unwrap();

    println!("Loaded [{inp_path}]");

    convert_from_bin::<MarlinFormat, ChessBoard>(inp_path, out_path.clone(), threads).unwrap();

    println!();

    println!("Written to [{out_path}]");
}
