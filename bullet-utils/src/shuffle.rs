use std::{fs::File, io::{BufWriter, Write}, path::PathBuf, time::Instant};

use bullet_core::Rand;
use bulletformat::ChessBoard;
use structopt::StructOpt;

#[derive(StructOpt)]
pub struct ShuffleOptions {
    #[structopt(required = true, short, long)]
    input: PathBuf,
    #[structopt(required = true, short, long)]
    output: PathBuf,
}

impl ShuffleOptions {
    pub fn run(&self) {
        let mut raw_bytes = std::fs::read(&self.input).unwrap();
        let data = to_slice_with_lifetime_mut(&mut raw_bytes);

        let mut output =
            BufWriter::new(File::create(&self.output).expect("Provide a correct path!"));

        println!("# [Shuffling Data]");
        let time = Instant::now();
        shuffle(data);
        println!("> Took {:.2} seconds.", time.elapsed().as_secs_f32());

        write_data(data, &mut output);
    }
}


fn to_slice_with_lifetime_mut<T, U>(slice: &mut [T]) -> &mut [U] {
    let src_size = std::mem::size_of_val(slice);
    let tgt_size = std::mem::size_of::<U>();

    assert!(
        src_size % tgt_size == 0,
        "Target type size does not divide slice size!"
    );

    let len = src_size / tgt_size;
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), len) }
}

fn write_data(data: &mut [ChessBoard], output: &mut BufWriter<File>) {
    if data.is_empty() {
        return;
    }

    let data_slice = to_slice_with_lifetime_mut(data);

    output
        .write_all(data_slice)
        .expect("Nothing can go wrong in unsafe code!");
}

fn shuffle(data: &mut [ChessBoard]) {
    let mut rng = Rand::default();

    for i in (0..data.len()).rev() {
        let idx = rng.rand_int() as usize % (i + 1);
        data.swap(idx, i);
    }
}
