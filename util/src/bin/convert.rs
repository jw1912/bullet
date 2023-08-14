use std::{
    env::args,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

use data::PackedPosition;

fn main() {
    let inp_path = args().nth(1).expect("Expected a file name!");
    let out_path = args().nth(2).expect("Expected a file name!");

    let file = BufReader::new(File::open(&inp_path).expect("Provide a correct path!"));

    println!("Loaded [{inp_path}]");

    let mut data = Vec::new();

    for line in file.lines().map(Result::unwrap) {
        data.push(PackedPosition::from_fen(&line));
    }

    println!("Parsed to PackedPosition");

    let mut output = BufWriter::new(File::create(&out_path).expect("Provide a correct path!"));

    let data_slice = unsafe { util::to_slice_with_lifetime(&data) };

    output
        .write_all(data_slice)
        .expect("Nothing can go wrong in unsafe code!");

    println!("Written to [{out_path}]");
}
