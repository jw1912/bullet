use std::{
    env::args,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use bullet::{data::MarlinFormat, util::to_slice_with_lifetime};

fn main() {
    let timer = Instant::now();

    let inp_path = args().nth(1).expect("Expected a file name!");
    let out_path = args().nth(2).expect("Expected a file name!");

    let file = BufReader::new(File::open(&inp_path).expect("Provide a correct path!"));

    println!("Loaded [{inp_path}]");

    let mut data = Vec::new();

    let mut results = [0, 0, 0];

    for line in file.lines().map(Result::unwrap) {
        match MarlinFormat::from_epd(&line) {
            Ok(pos) => {
                results[pos.result_idx()] += 1;
                data.push(pos);
            }
            Err(message) => println!("{message}"),
        }
    }

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

    let mut output = BufWriter::new(File::create(&out_path).expect("Provide a correct path!"));

    let data_slice = unsafe { to_slice_with_lifetime(&data) };

    output
        .write_all(data_slice)
        .expect("Nothing can go wrong in unsafe code!");

    println!("Written to [{out_path}]");
}

#[test]
fn test_parse() {
    let pos = MarlinFormat::from_epd("r1bq1bnr/pppp1kp1/2n1p3/5N1p/1PP5/8/P2PPPPP/RNBQKB1R w - - 0 1 55 [1.0]")
        .unwrap();

    let pieces = [
        "WHITE PAWN",
        "WHITE KNIGHT",
        "WHITE BISHOP",
        "WHITE ROOK",
        "WHITE QUEEN",
        "WHITE KING",
        "BLACK PAWN",
        "BLACK KNIGHT",
        "BLACK BISHOP",
        "BLACK ROOK",
        "BLACK QUEEN",
        "BLACK KING",
    ];

    let files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

    for (colour, piece, square) in pos {
        let pc = pieces[usize::from(colour * 6 + piece)];
        let sq = format!("{}{}", files[usize::from(square) % 8], 1 + square / 8);
        println!("{pc}: {sq}")
    }

    println!("{pos:#?}");

    println!("res: {}", pos.result());
    println!("stm: {}", pos.stm());
    println!("score: {}", pos.score());
}
