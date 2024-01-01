use std::{
    env::args,
    fs::File,
    io::{BufRead, BufReader, BufWriter},
    time::Instant,
};

use bulletformat::{BulletFormat, ChessBoard};

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
        match line.parse::<ChessBoard>() {
            Ok(pos) => {
                results[pos.result_idx()] += 1;
                data.push(pos);
            }
            Err(message) => println!("error parsing: {message}"),
        }

        if data.len() % 16384 == 0 {
            BulletFormat::write_to_bin(&mut output, &data).unwrap();
            data.clear();
        }
    }

    BulletFormat::write_to_bin(&mut output, &data).unwrap();

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
