use std::{
    env::args,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write}, time::Instant,
};

use data::Position;

fn main() {
    let timer = Instant::now();

    let inp_path = args().nth(1).expect("Expected a file name!");
    let out_path = args().nth(2).expect("Expected a file name!");

    let file = BufReader::new(File::open(&inp_path).expect("Provide a correct path!"));

    println!("Loaded [{inp_path}]");

    let mut data = Vec::new();

    let mut results = [0, 0, 0];

    for line in file.lines().map(Result::unwrap) {
        match Position::from_epd(&line) {
            Ok(pos) => {
                results[pos.result_idx()] += 1;
                data.push(pos);
            },
            Err(message) => println!("{message}"),
        }
    }

    println!("Parsed to Position");
    println!("Summary: {} Positions in {:.2} seconds", results.iter().sum::<u64>(), timer.elapsed().as_secs_f32());
    println!("Wins: {}, Draws: {}, Losses: {}", results[2], results[1], results[0]);

    let mut output = BufWriter::new(File::create(&out_path).expect("Provide a correct path!"));

    let data_slice = unsafe { data::util::to_slice_with_lifetime(&data) };

    output
        .write_all(data_slice)
        .expect("Nothing can go wrong in unsafe code!");

    println!("Written to [{out_path}]");
}
