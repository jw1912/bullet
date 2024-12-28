mod interleave;

use structopt::StructOpt;

#[derive(StructOpt)]
pub enum MontyBinpackOptions {
    Interleave(interleave::InterleaveOptions),
}

impl MontyBinpackOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        match self {
            Self::Interleave(options) => options.run(),
        }
    }
}
