use std::{
    env::args,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use bullet::{
    data::{marlinformat::MarlinFormat, ChessBoard},
    util::to_slice_with_lifetime,
};

fn main() {
    let timer = Instant::now();

    let inp_path = args().nth(1).expect("Expected a file name!");
    let out_path = args().nth(2).expect("Expected a file name!");

    println!("Loaded [{inp_path}]");

    let mut data = Vec::new();
    let mut count = 0;

    let cap = 128 * 16384 * std::mem::size_of::<MarlinFormat>();

    let mut file = BufReader::with_capacity(cap, File::open(&inp_path).unwrap());
    while let Ok(buf) = file.fill_buf() {
        if buf.is_empty() {
            break;
        }

        let consumed = buf.len();

        let new_buf = unsafe { to_slice_with_lifetime(buf) };

        let additional = new_buf.len();

        for mf in new_buf {
            data.push(ChessBoard::from_marlinformat(mf));
        }

        file.consume(consumed);

        count += additional;
        print!(
            "Positions: {count} ({:.0} pos/sec)\r",
            count as f64 / timer.elapsed().as_secs_f64()
        );
        let _ = std::io::stdout().flush();
    }

    println!();

    let mut output = BufWriter::new(File::create(&out_path).expect("Provide a correct path!"));

    let data_slice = unsafe { to_slice_with_lifetime(&data) };

    output
        .write_all(data_slice)
        .expect("Nothing can go wrong in unsafe code!");

    println!("Written to [{out_path}]");
}
