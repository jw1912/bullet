mod convert;
mod count_buckets;
mod interleave;
mod montybinpack;
mod shuffle;
mod validate;
mod viribinpack;

use oorandom::Rand64;
use structopt::StructOpt;

#[derive(StructOpt)]
pub enum Options {
    Convert(convert::ConvertOptions),
    Interleave(interleave::InterleaveOptions),
    Shuffle(shuffle::ShuffleOptions),
    Validate(validate::ValidateOptions),
    BucketCount(count_buckets::ValidateOptions),
    Montybinpack(montybinpack::MontyBinpackOptions),
    Viribinpack(viribinpack::ViriBinpackOptions),
}

fn main() -> anyhow::Result<()> {
    match Options::from_args() {
        Options::Convert(options) => options.run(),
        Options::Interleave(options) => options.run(),
        Options::Shuffle(options) => options.run(),
        Options::Validate(options) => options.run(),
        Options::BucketCount(options) => options.run(),
        Options::Montybinpack(options) => options.run(),
        Options::Viribinpack(options) => options.run(),
    }
}

fn seeded_rng() -> Rand64 {
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("valid").as_nanos();

    Rand64::new(seed)
}
