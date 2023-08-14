use std::{env::args, io::{BufRead, BufReader, BufWriter, Write}, fs::File};

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

    let data_slice = unsafe { slice_with_lifetime(&data) };

    output.write_all(data_slice).expect("Nothing can go wrong in unsafe code!");

    println!("Written to [{out_path}]");
}

/// # Safety
/// They're just bytes, they can hold anything.
unsafe fn slice_with_lifetime<T>(slice: &[T]) -> &[u8] {
    let len = std::mem::size_of::<PackedPosition>() * slice.len();
    std::slice::from_raw_parts(slice.as_ptr().cast(), len)
}