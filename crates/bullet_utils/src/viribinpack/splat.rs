use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
};

use anyhow::Context;
use bulletformat::ChessBoard;
use structopt::StructOpt;
use viriformat::dataformat::{Filter, Game};

#[derive(StructOpt)]
pub struct SplatOptions {
    /// Path to input viriformat file.
    #[structopt(required = true)]
    pub input: PathBuf,
    /// Path to output bulletformat file.
    #[structopt(required = true)]
    pub output: PathBuf,
    /// Optional path to a viriformat filter config toml.
    pub cfg: PathBuf,
}

impl SplatOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        println!("Reading from [{:#?}]", self.input);

        let input = File::open(&self.input).with_context(|| format!("Failed to open {}", self.input.display()))?;
        let output =
            File::create(&self.output).with_context(|| format!("Failed to create {}", self.output.display()))?;
        let bytes = input.metadata()?.len();

        let mut reader = BufReader::new(input);
        let mut writer = BufWriter::new(output);
        let mut games = 0usize;
        let mut positions = 0usize;

        let filter = Filter::from_path(&self.cfg).unwrap_or(Filter::UNRESTRICTED);

        let mut buffer = Vec::new();

        while let Ok(game) = Game::deserialise_from(&mut reader, buffer) {
            games += 1;
            positions += game.moves.len();

            game.splat_to_bulletformat(
                |bf_board| {
                    let bytes = unsafe { std::mem::transmute::<ChessBoard, [u8; 32]>(bf_board) };
                    writer.write_all(&bytes)?;
                    Ok(())
                },
                &filter,
            )?;

            if games % 16384 == 0 {
                print!("Splatted {games} games\r");
            }

            buffer = game.moves;
            buffer.clear();
        }

        println!();
        println!("Summary:");
        println!("Games = {games}");
        println!("Positions = {positions}");
        println!("Bytes per position = {}", bytes as f64 / positions as f64);

        Ok(())
    }
}
