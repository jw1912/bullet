use anyhow::Context;
use bulletformat::{ChessBoard, DataLoader};
use structopt::StructOpt;

use std::{path::PathBuf, time::Instant};

#[derive(StructOpt)]
pub struct ValidateOptions {
    #[structopt(required = true, short, long)]
    input: PathBuf,
}

impl ValidateOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        let loader = DataLoader::<ChessBoard>::new(&self.input, 256).with_context(|| "Failed to create dataloader.")?;

        let mut done = 0usize;
        let timer = Instant::now();
        let mut results = [0u64; 3];

        let mut invalid = [0u64; 6];
        let mut found = false;

        let mut check = |cond: bool, idx| {
            if !cond {
                invalid[idx] += 1;

                if !found {
                    found = true;
                    println!("There is at least one invalid position!");
                }
            }
        };

        loader.map_positions(|pos| {
            let mut counts = [0; 12];

            for (piece, square) in pos.into_iter() {
                let pc = usize::from(piece & 7);
                let c = usize::from(piece >> 3);

                counts[6 * c + pc] += 1;

                if pc == 5 {
                    if c == 0 {
                        check(pos.our_ksq() == square, 4);
                    } else {
                        check(pos.opp_ksq() == square ^ 56, 4);
                    }
                } else if pc == 0 {
                    check(![0, 7].contains(&(square / 8)), 5);
                }
            }

            let total = counts.iter().sum::<i32>();
            check(counts[5] == 1, 0);
            check(counts[11] == 1, 1);
            check(total > 2, 2);
            check(total <= 32, 3);

            results[usize::from(pos.result)] += 1;

            done += 1;
            if done % 10_000_000 == 0 {
                println!("Checked {done} Positions")
            }
        });

        let msgs = [
            "Invalid number of stm kings",
            "Invalid number of nstm kings",
            "No non-king pieces on the board",
            "Too many pieces on the board",
            "King square does not match occupancy",
            "Pawn on 1st/8th rank",
        ];

        println!();
        println!("SUMMARY:");

        let total = results.iter().sum::<u64>();
        println!("Checked {total} Positions in {:.2} seconds", timer.elapsed().as_secs_f32());

        let w = results[2] * 100 / total;
        let d = results[1] * 100 / total;
        let l = results[0] * 100 / total;
        println!("Wins: {w}%, Draws: {d}%, Losses: {l}%");

        let total_invalid = invalid.iter().sum::<u64>();
        if total_invalid > 0 {
            println!();
            println!("ERRORS:");
            for (&count, msg) in invalid.iter().zip(msgs.iter()) {
                println!("{msg: <35} : {count}");
            }
            println!("--------------------------------");
            println!("Total errors: {total_invalid}");
            println!("Note this is total errors, 1 position may contribute multiple.")
        } else {
            println!("No invalid positions!")
        }

        Ok(())
    }
}
