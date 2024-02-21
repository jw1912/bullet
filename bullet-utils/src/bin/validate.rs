use bulletformat::{ChessBoard, DataLoader};

fn main() {
    let inp_path = std::env::args().nth(1).expect("Expected a file name!");
    let loader = DataLoader::<ChessBoard>::new(inp_path, 256).unwrap();

    let mut done = 0;

    loader.map_positions(|pos| {
        let mut count = 0;
        for (piece, square) in pos.into_iter() {
            let pc = usize::from(piece & 7);
            let c = usize::from(piece >> 3);

            if pc == 5 {
                count += 1;

                if c == 0 && pos.our_ksq() != square {
                    panic!("Invalid King Square!");
                }

                if c == 1 && pos.opp_ksq() != square ^ 56 {
                    panic!("Invalid King Square!");
                }
            }
        }

        assert_eq!(count, 2, "Invalid number of kings!");

        done += 1;
        if done % 10_000_000 == 0 {
            println!("Checked {done} positions.")
        }
    });
}