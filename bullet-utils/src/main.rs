mod convert;
mod interleave;
mod shuffle;
mod validate;

use structopt::StructOpt;

#[derive(StructOpt)]
pub enum Options {
    Convert(convert::ConvertOptions),
    Interleave(interleave::InterleaveOptions),
    Shuffle(shuffle::ShuffleOptions),
    Validate(validate::ValidateOptions),
}

fn main() {
    match Options::from_args() {
        Options::Convert(options) => options.run(),
        Options::Interleave(options) => options.run(),
        Options::Shuffle(options) => options.run(),
        Options::Validate(options) => options.run(),
    }
}
