use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context};
use bulletformat::{
    chess::{CudADFormat, MarlinFormat},
    convert_from_bin, convert_from_text, AtaxxBoard, BulletFormat, ChessBoard,
};
use structopt::StructOpt;

#[derive(StructOpt)]
pub struct ConvertOptions {
    #[structopt(required = true, short, long)]
    from: String,
    #[structopt(required = true, short, long)]
    input: PathBuf,
    #[structopt(required = true, short, long)]
    output: PathBuf,
    #[structopt(short, long, default_value = "1")]
    threads: usize,
}

impl ConvertOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        match self.from.as_str() {
            "marlinformat" => convert_from_bin::<MarlinFormat, ChessBoard>(&self.input, &self.output, self.threads)
                .with_context(|| "Failed to convert marlinformat."),
            "cudadformat" => convert_from_bin::<CudADFormat, ChessBoard>(&self.input, &self.output, self.threads)
                .with_context(|| "Failed to convert cudadformat."),
            "text" => convert_text(&self.input, &self.output),
            "ataxx" => convert_from_text::<AtaxxBoard>(&self.input, &self.output)
                .with_context(|| "Failed to convert ataxxformat."),
            _ => bail!("Unrecognised Source Type! Supported: 'marlinformat', 'text', 'ataxx'."),
        }
    }
}

fn convert_text(inp_path: impl AsRef<Path>, out_path: impl AsRef<Path>) -> anyhow::Result<()> {
    let timer = Instant::now();

    let file = BufReader::new(File::open(&inp_path).with_context(|| "Provide a correct path!")?);

    let mut data = Vec::new();

    let mut results = [0, 0, 0];

    let mut output = BufWriter::new(File::create(&out_path).with_context(|| "Provide a correct path!")?);

    for line in file.lines() {
        match line?.parse::<ChessBoard>() {
            Ok(pos) => {
                results[pos.result_idx()] += 1;
                data.push(pos);
            }
            Err(message) => println!("error parsing: {message}"),
        }

        if data.len() % 16384 == 0 {
            BulletFormat::write_to_bin(&mut output, &data).with_context(|| "Failed to write boards into output.")?;
            data.clear();
        }
    }

    BulletFormat::write_to_bin(&mut output, &data).with_context(|| "Failed to write boards into output.")?;

    println!("Parsed to Position");
    println!("Summary: {} Positions in {:.2} seconds", results.iter().sum::<u64>(), timer.elapsed().as_secs_f32());
    println!("Wins: {}, Draws: {}, Losses: {}", results[2], results[1], results[0]);

    Ok(())
}
