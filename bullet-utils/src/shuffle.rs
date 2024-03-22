use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Result, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use bullet::{util, Rand};
use bulletformat::ChessBoard;
use structopt::StructOpt;

use crate::interleave::InterleaveOptions;

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
    pub fn run(&self) {
        let input_size = fs::metadata(self.input.clone()).expect("Input file is valid").len() as usize;
        assert_eq!(0, input_size % CHESS_BOARD_SIZE);

        println!("# [Shuffling Data]");
        let time = Instant::now();

        if input_size <= self.mem_used_mb * BYTES_PER_MB {
            let mut raw_bytes = std::fs::read(&self.input).unwrap();
            let data = util::to_slice_with_lifetime_mut(&mut raw_bytes);

            shuffle_positions(data);

            let mut output = BufWriter::new(File::create(&self.output).expect("Provide a correct path!"));

            write_data(data, &mut output);
        } else {
            let temp_dir = Path::new(TMP_DIR);
            if !Path::exists(temp_dir) {
                fs::create_dir(temp_dir).expect("Temp dir could not be created.");
            }
            let bytes_used = self.mem_used_mb * BYTES_PER_MB;
            let num_tmp_files = ((input_size + bytes_used - 1) / bytes_used).max(MIN_TMP_FILES);
            let temp_files = (0..num_tmp_files)
                .map(|idx| {
                    let output_file = format!("{}/part_{}.bin", temp_dir.to_str().unwrap(), idx + 1);
                    PathBuf::from(output_file)
                })
                .collect::<Vec<_>>();

            assert!(self.split_file(&temp_files, input_size).is_ok());

            println!("# [Finished splitting data. Interleaving...]");
            let interleave = InterleaveOptions::new(temp_files.to_vec(), self.output.clone());
            interleave.run();

            if fs::remove_dir_all(temp_dir).is_err() {
                println!("Error automatically removing temp files");
            }
        }

        println!("> Took {:.2} seconds.", time.elapsed().as_secs_f32());
    }

    fn split_file(&self, temp_files: &[PathBuf], input_size: usize) -> Result<()> {
        let mut input = BufReader::new(File::open(self.input.clone()).unwrap());
        let temp_files =
            temp_files.iter().map(|f| File::create(f).expect("Tmp file could not be created.")).collect::<Vec<_>>();

        let total_positions = input_size / CHESS_BOARD_SIZE;
        let ideal_positions_per_file = total_positions / temp_files.len();
        let mut positions_per_file = vec![ideal_positions_per_file; temp_files.len()];
        let remaining_positions = total_positions % temp_files.len();
        for size in positions_per_file.iter_mut().take(remaining_positions) {
            *size += 1;
        }

        for (idx, file) in temp_files.iter().enumerate() {
            println!("# [Shuffling temp file {} / {}]", idx + 1, temp_files.len());
            println!("    -> Reading into ram");
            let buffer_size = positions_per_file[idx] * CHESS_BOARD_SIZE;
            let mut buffer = vec![0u8; buffer_size];
            input.read_exact(&mut buffer[0..buffer_size])?;

            println!("    -> Shuffling in memory");
            let data = util::to_slice_with_lifetime_mut(&mut buffer[0..buffer_size]);
            shuffle_positions(data);
            let data_slice = util::to_slice_with_lifetime(data);
            assert_eq!(0, buffer_size % CHESS_BOARD_SIZE);

            println!("    -> Writing to temp file");
            let mut writer = BufWriter::new(file);
            writer.write_all(data_slice)?;
        }

        Ok(())
    }
}

fn shuffle_positions(data: &mut [ChessBoard]) {
    let mut rng = Rand::default();

    for i in (0..data.len()).rev() {
        let idx = rng.rand_int() as usize % (i + 1);
        data.swap(idx, i);
    }
}

fn write_data(data: &[ChessBoard], output: &mut BufWriter<File>) {
    if data.is_empty() {
        return;
    }

    let data_slice = util::to_slice_with_lifetime(data);

    output.write_all(data_slice).expect("Nothing can go wrong in unsafe code!");
}
