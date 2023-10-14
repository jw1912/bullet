use common::{data::{ChessBoard, Features}, HIDDEN};
use cpu::{Accumulator, NetworkParams};

const FENS: [&str; 2] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
];

fn main() {
    let name = std::env::args().nth(1).expect("Expects a valid name.");

    println!("DISCLAIMER: Assumes a scale of 400.");

    println!("Loading [{name}]");

    let net_path = format!("checkpoints/{name}/params.bin");

    println!("Path [{net_path}]");

    let mut params = NetworkParams::new();
    params.load_from_bin(&net_path);

    for fen in FENS {
        let epd = format!("{fen} | 0 | 0.0");

        let bin_fmt = ChessBoard::from_epd(&epd).unwrap();

        let bias = Accumulator::load_biases(&params);
        let mut accs = [bias; 2];
        let mut activated = [[0.0; HIDDEN]; 2];
        let mut features = Features::default();

        let (eval, _) = params.forward(&bin_fmt, &mut accs, &mut activated, &mut features);

        println!("FEN: {fen}");
        println!("EVAL: {}", eval * 400.0);
        println!();
    }
}