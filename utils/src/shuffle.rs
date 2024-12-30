use std::{
    fs::{self, File},
    io::{BufReader, IoSliceMut, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Context;
use bulletformat::ChessBoard;
use structopt::StructOpt;

use crate::{interleave::InterleaveOptions, Rand};

#[derive(StructOpt)]
pub struct ShuffleOptions {
    #[structopt(required = true, short, long)]
    pub input: PathBuf,
    #[structopt(required = true, short, long)]
    pub output: PathBuf,
    #[structopt(required = true, short, long)]
    pub mem_used_mb: usize,
}

const CHESS_BOARD_SIZE: usize = std::mem::size_of::<ChessBoard>();
const MIN_TMP_FILES: usize = 4;
const BYTES_PER_MB: usize = 1_048_576;
const TMP_DIR: &str = "./tmp";

impl ShuffleOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        let input_size = fs::metadata(self.input.clone()).with_context(|| "Input file is invalid.")?.len() as usize;
        assert_eq!(0, input_size % CHESS_BOARD_SIZE);

        // Test path before doing useless work
        validate_output_path(Path::new(&self.output))
            .with_context(|| format!("Invalid output path: {}", self.output.display()))?;

        println!("# [Shuffling Data]");
        let time = Instant::now();

        if input_size <= self.mem_used_mb * BYTES_PER_MB {
            let mut raw_bytes = std::fs::read(&self.input).with_context(|| "Failed to read input.")?;

            shuffle_positions(&mut raw_bytes);

            let mut file = File::create(&self.output).with_context(|| "Provide a correct path!")?;
            file.write_all(&raw_bytes)?;
        } else {
            let temp_dir = Path::new(TMP_DIR);
            if !Path::exists(temp_dir) {
                fs::create_dir(temp_dir).with_context(|| "Temp dir could not be created.")?;
            }
            let bytes_used = self.mem_used_mb * BYTES_PER_MB;
            let num_tmp_files = input_size.div_ceil(bytes_used).max(MIN_TMP_FILES);
            let temp_files = (0..num_tmp_files)
                .map(|idx| {
                    let output_file = format!(
                        "{}/part_{}.bin",
                        temp_dir.to_str().with_context(|| "Failed to convert path to string.")?,
                        idx + 1
                    );
                    Ok(PathBuf::from(output_file))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            assert!(self.split_file(&temp_files, input_size).is_ok());

            println!("# [Finished splitting data. Interleaving...]");
            let interleave = InterleaveOptions::new(temp_files.to_vec(), self.output.clone());
            interleave.run()?;

            if fs::remove_dir_all(temp_dir).is_err() {
                println!("Error automatically removing temp files");
            }
        }

        println!("> Took {:.2} seconds.", time.elapsed().as_secs_f32());

        Ok(())
    }

    fn split_file(&self, temp_files: &[PathBuf], input_size: usize) -> anyhow::Result<()> {
        let mut input = BufReader::new(File::open(self.input.clone()).with_context(|| "Failed to open file.")?);
        let temp_files = temp_files
            .iter()
            .map(|f| File::create(f).with_context(|| "Tmp file could not be created."))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let total_positions = input_size / CHESS_BOARD_SIZE;
        let ideal_positions_per_file = total_positions / temp_files.len();
        let mut positions_per_file = vec![ideal_positions_per_file; temp_files.len()];
        let remaining_positions = total_positions % temp_files.len();
        for size in positions_per_file.iter_mut().take(remaining_positions) {
            *size += 1;
        }

        for (idx, mut file) in temp_files.iter().enumerate() {
            println!("# [Shuffling temp file {} / {}]", idx + 1, temp_files.len());
            println!("    -> Reading into ram");

            let buffer_size = positions_per_file[idx] * CHESS_BOARD_SIZE;
            let mut buffer = vec![0u8; buffer_size];

            // performs better than a read_exact

            let chunk_size = 1024 * 1024;
            let mut offset = 0;

            while offset < buffer_size {
                let remaining = buffer_size - offset;
                let current_chunk = remaining.min(chunk_size);
                let mut iovec = [IoSliceMut::new(&mut buffer[offset..offset + current_chunk])];
                let bytes_read = input.read_vectored(&mut iovec)?;

                if bytes_read == 0 {
                    break;
                }

                offset += bytes_read;
            }

            println!("    -> Shuffling in memory");

            shuffle_positions(&mut buffer[0..buffer_size]);

            println!("    -> Writing to temp file");
            file.write_all(&buffer[0..buffer_size])?;
        }

        Ok(())
    }
}

fn shuffle_positions(data: &mut [u8]) {
    assert_eq!(data.len() % CHESS_BOARD_SIZE, 0);

    let len = data.len() / CHESS_BOARD_SIZE;

    let mut rng = Rand::default();

    for i in (0..len).rev() {
        let idx = rng.rand() as usize % (i + 1);
        for j in 0..CHESS_BOARD_SIZE {
            data.swap(CHESS_BOARD_SIZE * idx + j, CHESS_BOARD_SIZE * i + j);
        }
    }
}

/// Test if we can write to the output path
fn validate_output_path(path: &Path) -> anyhow::Result<()> {
    match File::create(path) {
        Ok(_) => Ok(()),
        Err(e) => Err(anyhow::anyhow!("Cannot create file at specified path: {}", e)),
    }
}
