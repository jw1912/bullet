use std::{
    env::args,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use common::{data::ChessBoard, util::to_slice_with_lifetime};

fn main() {
    let timer = Instant::now();

    let inp_path = args().nth(1).expect("Expected a file name!");
    let out_path = args().nth(2).expect("Expected a file name!");

    let file = BufReader::new(File::open(&inp_path).expect("Provide a correct path!"));

    println!("Loaded [{inp_path}]");

    let mut data = Vec::new();

    let mut results = [0, 0, 0];

    let mut output = BufWriter::new(File::create(&out_path).expect("Provide a correct path!"));

    for line in file.lines().map(Result::unwrap) {
        match ChessBoard::from_epd(&line) {
            Ok(pos) => {
                results[pos.result_idx()] += 1;
                data.push(pos);
            }
            Err(message) => println!("{message}"),
        }

        if data.len() % 16384 == 0 {
            write(&mut data, &mut output);
        }
    }

    write(&mut data, &mut output);

    println!("Parsed to Position");
    println!(
        "Summary: {} Positions in {:.2} seconds",
        results.iter().sum::<u64>(),
        timer.elapsed().as_secs_f32()
    );
    println!(
        "Wins: {}, Draws: {}, Losses: {}",
        results[2], results[1], results[0]
    );

    println!("Written to [{out_path}]");
}

fn write(data: &mut Vec<ChessBoard>, output: &mut BufWriter<File>) {
    if data.is_empty() {
        return
    }

    let data_slice = unsafe { to_slice_with_lifetime(data) };

    output
        .write_all(data_slice)
        .expect("Nothing can go wrong in unsafe code!");

    data.clear();
}
