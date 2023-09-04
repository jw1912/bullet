use std::{
    env::args,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use bullet::{position::Position, util::to_slice_with_lifetime};

fn from_epd(fen: &str) -> Result<Position, String> {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let board_str = parts[0];
    let stm_str = parts[1];

    let mut occ = 0;
    let mut pcs = [0; 16];

    let mut idx = 0;
    for (i, row) in board_str.split('/').rev().enumerate() {
        let mut col = 0;
        for ch in row.chars() {
            if ('1'..='8').contains(&ch) {
                col += ch.to_digit(10).expect("hard coded") as usize;
            } else if let Some(piece) = "PNBRQKpnbrqk".chars().position(|el| el == ch) {
                let square = 8 * i + col;
                occ |= 1 << square;
                let pc = (piece % 6) | (piece / 6) << 3;
                pcs[idx / 2] |= (pc as u8) << (4 * (idx & 1));
                idx += 1;
                col += 1;
            }
        }
    }

    // don't currently worry about en passant square
    let stm_enp = u8::from(stm_str == "b") << 7;

    let hfm = parts[4].parse().unwrap_or(0);

    let fmc = parts[5].parse().unwrap_or(1);

    let score = parts[6].parse::<i16>().unwrap_or(0);

    let result = match parts[7] {
        "[1.0]" => 2,
        "[0.5]" => 1,
        "[0.0]" => 0,
        _ => {
            println!("{fen}");
            return Err(String::from("Bad game result!"));
        }
    };

    Ok(Position::new(occ, pcs, stm_enp, hfm, fmc, score, result, 0))
}

fn main() {
    let timer = Instant::now();

    let inp_path = args().nth(1).expect("Expected a file name!");
    let out_path = args().nth(2).expect("Expected a file name!");

    let file = BufReader::new(File::open(&inp_path).expect("Provide a correct path!"));

    println!("Loaded [{inp_path}]");

    let mut data = Vec::new();

    let mut results = [0, 0, 0];

    for line in file.lines().map(Result::unwrap) {
        match from_epd(&line) {
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
    let pos = from_epd("r1bq1bnr/pppp1kp1/2n1p3/5N1p/1PP5/8/P2PPPPP/RNBQKB1R w - - 0 1 55 [1.0]")
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
