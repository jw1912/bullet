use std::{fs::File, io::BufReader, path::PathBuf};

use structopt::StructOpt;
use viriformat::dataformat::Game;

#[derive(StructOpt)]
pub struct CountOptions {
    #[structopt(required = true)]
    pub input: PathBuf,
}

impl CountOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        println!("Reading from [{:#?}]", self.input);

        let file = File::open(&self.input)?;
        let bytes = file.metadata()?.len();

        let mut reader = BufReader::new(file);
        let mut games = 0usize;
        let mut positions = 0usize;

        let mut buffer = Vec::new();

        while let Ok(game) = Game::deserialise_from(&mut reader, buffer) {
            games += 1;
            positions += game.moves.len();

            if games % 16384 == 0 {
                print!("Counted {games} games\r");
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
