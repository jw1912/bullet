use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
};

use crate::Rand;
use anyhow::Context;
use structopt::StructOpt;

#[derive(StructOpt)]
pub struct InterleaveOptions {
    #[structopt(required = true, min_values = 2)]
    pub inputs: Vec<PathBuf>,
    #[structopt(required = true, short, long)]
    pub output: PathBuf,
}

impl InterleaveOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        const SIZE: usize = 32;

        println!("Writing to {:#?}", self.output);
        println!("Reading from:\n{:#?}", self.inputs);
        let mut streams = Vec::new();
        let mut total = 0;

        let target = File::create(&self.output).with_context(|| "Failed to create output file")?;
        let mut writer = BufWriter::new(target);

        for path in &self.inputs {
            let file =
                File::open(path).with_context(|| format!("Failed to open {path}", path = path.to_string_lossy()))?;
            let count = file.metadata()?.len() as usize / SIZE;

            if count > 0 {
                streams.push((count, BufReader::new(file)));
                total += count;
            }
        }

        let mut remaining = total;
        let mut rng = Rand::default();

        while remaining > 0 {
            let mut spot = rng.rand() as usize % remaining;
            let mut idx = 0;
            while streams[idx].0 < spot {
                spot -= streams[idx].0;
                idx += 1;
            }

            let (count, reader) = &mut streams[idx];
            let mut value = [0; SIZE];
            reader.read_exact(&mut value)?;
            writer.write_all(&value)?;

            remaining -= 1;
            *count -= 1;
            if *count == 0 {
                streams.swap_remove(idx);
            }

            if remaining % 1_048_576 == 0 {
                let written = total - remaining;
                print!("Written {written} / {total} ({:.2})\r", written as f32 / total as f32 * 100.0);
                let _ = std::io::stdout().flush();
            }
        }

        Ok(())
    }

    pub fn new(inputs: Vec<PathBuf>, output: PathBuf) -> Self {
        Self { inputs, output }
    }
}
