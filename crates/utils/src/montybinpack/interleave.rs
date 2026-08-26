use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use montyformat::{FastDeserialise, MontyValueFormat};
use structopt::StructOpt;

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

        for path in &self.inputs {
            let file = File::open(path)?;

            let count = count_games(path)?;

            if count > 0 {
                streams.push((count, BufReader::new(file)));
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

            let (count, reader) = &mut streams[idx];

            MontyValueFormat::deserialise_fast_into_buffer(reader, &mut buffer)?;
            writer.write_all(&buffer)?;

            remaining -= 1;
            *count -= 1;
            if *count == 0 {
                streams.swap_remove(idx);
            }

            if remaining / INTERVAL < prev {
                prev = remaining / INTERVAL;
                let written = total - remaining;
                print!("Written {written}/{total} Games ({:.2}%)\r", written as f64 / total as f64 * 100.0);
                let _ = std::io::stdout().flush();
            }
        }

        println!();
        println!("Written {total} games to {:#?}", self.output);

        Ok(())
    }
}

fn count_games(path: &Path) -> anyhow::Result<u64> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut buffer = Vec::new();
    let mut games = 0;

    while MontyValueFormat::deserialise_fast_into_buffer(&mut reader, &mut buffer).is_ok() {
        games += 1;
    }

    println!("{path:#?} contains {games} games");

    Ok(games)
}
