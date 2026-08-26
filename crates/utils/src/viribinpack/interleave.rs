use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use structopt::StructOpt;
use viriformat::dataformat::Game;

use crate::seeded_rng;

#[derive(StructOpt)]
pub struct InterleaveOptions {
    #[structopt(required = true, min_values = 2)]
    pub inputs: Vec<PathBuf>,
    #[structopt(required = true, short, long)]
    pub output: PathBuf,
}

impl InterleaveOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        println!("Writing to {:#?}", self.output);
        println!("Reading from:\n{:#?}", self.inputs);
        let mut streams = Vec::new();
        let mut total = 0;

        let target = File::create(&self.output)?;
        let mut writer = BufWriter::new(target);

        let mut total_input_file_size = 0;
        for path in &self.inputs {
            let file = File::open(path)?;

            total_input_file_size += file.metadata()?.len();

            let count = count_games(path)?;

            if count > 0 {
                let fname =
                    path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "<unknown>".into());
                streams.push((count, BufReader::new(file), fname));
                total += count;
            }
        }

        let mut remaining = total;
        let mut rng = seeded_rng();

        const INTERVAL: u64 = 1024 * 1024;
        let mut prev = remaining / INTERVAL;

        let mut buffer = Vec::new();

        while remaining > 0 {
            let mut spot = rng.rand_range(0..remaining);
            let mut idx = 0;
            while streams[idx].0 <= spot {
                spot -= streams[idx].0;
                idx += 1;
            }

            let (count, reader, _) = &mut streams[idx];

            buffer.clear();
            Game::deserialise_fast_into_buffer(reader, &mut buffer)?;
            writer.write_all(&buffer)?;

            remaining -= 1;
            *count -= 1;
            if *count == 0 {
                println!("Finished reading {}", streams[idx].2);
                streams.swap_remove(idx);
            }

            if remaining / INTERVAL < prev {
                prev = remaining / INTERVAL;
                let written = total - remaining;
                print!("Written {written}/{total} Games ({:.2}%)\r", written as f64 / total as f64 * 100.0);
                let _ = std::io::stdout().flush();
            }
        }

        writer.flush()?;

        println!();
        println!("Written {total} games to {:#?}", self.output);

        let output_file = File::open(&self.output)?;
        let output_file_size = output_file.metadata()?.len();
        if output_file_size != total_input_file_size {
            anyhow::bail!("Output file size {output_file_size} does not match input file size {total_input_file_size}");
        }

        Ok(())
    }
}

fn count_games(path: &Path) -> anyhow::Result<u64> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut buffer = Vec::new();
    let mut games = 0;

    while Game::deserialise_fast_into_buffer(&mut reader, &mut buffer).is_ok() {
        buffer.clear();
        games += 1;
    }

    println!("{path:#?} contains {games} games");

    Ok(games)
}
