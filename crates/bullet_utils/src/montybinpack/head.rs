use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
};

use montyformat::{FastDeserialise, MontyValueFormat};
use structopt::StructOpt;

#[derive(StructOpt)]
pub struct HeadOptions {
    #[structopt(required = true)]
    pub input: PathBuf,
    #[structopt(required = true, short, long)]
    pub output: PathBuf,
    #[structopt(required = true, short, long)]
    pub games: usize,
}

impl HeadOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        println!("Writing to [{:#?}]", self.output);
        println!("Reading from [{:#?}]", self.input);

        let mut reader = BufReader::new(File::open(&self.input)?);
        let mut writer = BufWriter::new(File::create(&self.output)?);

        let mut buffer = Vec::new();
        let mut games = 0usize;
        let total = self.games;

        while MontyValueFormat::deserialise_fast_into_buffer(&mut reader, &mut buffer).is_ok() {
            writer.write_all(&buffer)?;
            buffer.clear();

            games += 1;

            if games % 16384 == 0 {
                print!("Written {games} / {total} ({:.2}%)\r", games as f64 / total as f64 * 100.0);
            }

            if games == total {
                break;
            }
        }

        println!("Written {games} / {total} ({:.2}%)", games as f64 / total as f64 * 100.0);

        Ok(())
    }
}
