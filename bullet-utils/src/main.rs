mod convert;
mod count_buckets;
mod interleave;
mod shuffle;
mod validate;
mod graph;

use structopt::StructOpt;

#[derive(StructOpt)]
pub enum Options {
    Convert(convert::ConvertOptions),
    Interleave(interleave::InterleaveOptions),
    Shuffle(shuffle::ShuffleOptions),
    Validate(validate::ValidateOptions),
    BucketCount(count_buckets::ValidateOptions),
    Graph(graph::GraphOptions),
}

fn main() {
    match Options::from_args() {
        Options::Convert(options) => options.run(),
        Options::Interleave(options) => options.run(),
        Options::Shuffle(options) => options.run(),
        Options::Validate(options) => options.run(),
        Options::BucketCount(options) => options.run(),
        Options::Graph(options) => options.run(),
    }
}

pub struct Rand(u32);
impl Default for Rand {
    fn default() -> Self {
        Self(
            (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("valid").as_nanos()
                & 0xFFFF_FFFF) as u32,
        )
    }
}

impl Rand {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    pub fn rand(&mut self, max: f64) -> f32 {
        let x = self.rand_int();
        ((0.5 - f64::from(x) / f64::from(u32::MAX)) * max * 2.0) as f32
    }

    pub fn rand_int(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}
