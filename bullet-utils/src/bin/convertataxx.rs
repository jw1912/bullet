use std::env::args;

use bulletformat::{convert_from_text, AtaxxBoard};

fn main() -> std::io::Result<()> {
    let inp_path = args().nth(1).expect("Expected a file name!");
    let out_path = args().nth(2).expect("Expected a file name!");

    convert_from_text::<AtaxxBoard>(inp_path, out_path)
}
