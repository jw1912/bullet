use anyhow::Context;
use bulletformat::{ChessBoard, DataLoader};
use structopt::StructOpt;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(StructOpt)]
pub struct ValidateOptions {
    #[structopt(required = true, min_values = 1)]
    pub inputs: Vec<PathBuf>,
    #[structopt(required = true, short, long)]
    bucket_file: String,
}

impl ValidateOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        let file = File::open(&self.bucket_file).with_context(|| "Couldn't find the bucket file!")?;
        let reader = BufReader::new(file);

        let mut numbers: Vec<usize> = Vec::new();
        for line in reader.lines() {
            let line = match line {
                Ok(line) => line,
                Err(e) => {
                    eprintln!("Error reading line: {e}");
                    continue;
                }
            };

            for token in line.replace(',', "").split(' ') {
                match token.trim().parse::<usize>() {
                    Ok(num) => numbers.push(num),
                    Err(e) => {
                        eprintln!("Error parsing number: {e}");
                        continue;
                    }
                }
            }
        }

        let mut num_buckets = 0;
        let mut buckets: [usize; 64] = [0; 64];
        for n in 0..numbers.len() {
            buckets[n] = numbers[n];
            num_buckets = num_buckets.max(numbers[n]);
        }

        println!("Bucket layout:");
        print_board(buckets);

        let mut total_position_count = 0;
        let mut total_king_squares: [usize; 64] = [0; 64];
        let mut total_bucket_counts: [usize; 64] = [0; 64];

        for path in &self.inputs {
            println!("\nFile {}", path.display());
            let loader = DataLoader::<ChessBoard>::new(path, 256).with_context(|| "Failed to create dataloader.")?;

            let mut position_count = 0usize;
            let mut king_squares: [usize; 64] = [0; 64];
            let mut bucket_counts: [usize; 64] = [0; 64];

            loader.map_positions(|pos| {
                let sq = usize::from(pos.our_ksq());
                position_count += 1;
                king_squares[sq] += 1;
                bucket_counts[buckets[sq]] += 1;
            });

            println!("King bucket distribution from {position_count} positions:");
            print_buckets(bucket_counts, num_buckets);

            total_position_count += position_count;
            for i in 0..64 {
                total_king_squares[i] += king_squares[i];
                total_bucket_counts[i] += bucket_counts[i];
            }
        }

        if self.inputs.len() != 1 {
            println!("\nTotal King bucket distribution from {total_position_count} positions:");
            print_buckets(total_bucket_counts, num_buckets);
        }

        println!("\nTotal king square counts:");
        print_board(total_king_squares);

        Ok(())
    }
}

pub fn print_buckets(arr: [usize; 64], num_buckets: usize) {
    for (bucket, count) in arr.iter().enumerate().take(num_buckets + 1) {
        println!("Bucket {bucket}: {count}");
    }
}

pub fn print_board(arr: [usize; 64]) {
    println!(
        "+-------------+------------+------------+------------+------------+------------+------------+------------+"
    );
    for y in (0..8).rev() {
        print!("| ");
        for x in 0..8 {
            let cnt = arr[(y * 8) + x];
            print!("{: >11} |", cnt.to_string())
        }
        println!("\n+-------------+------------+------------+------------+------------+------------+------------+------------+");
    }
}
