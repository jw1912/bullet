mod convert;
mod count_buckets;
mod interleave;
mod montybinpack;
mod shuffle;
mod validate;
mod viribinpack;

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

struct Rand(u64);

impl Default for Rand {
    fn default() -> Self {
        Self(
            (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("valid").as_nanos()
                & 0xFFFF_FFFF_FFFF_FFFF) as u64,
        )
    }
}

impl Rand {
    fn rand(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
