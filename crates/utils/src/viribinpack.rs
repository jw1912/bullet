mod count;
mod head;
mod interleave;
mod splat;

use structopt::StructOpt;

#[derive(StructOpt)]
pub enum ViriBinpackOptions {
    Head(head::HeadOptions),
    Interleave(interleave::InterleaveOptions),
    Count(count::CountOptions),
    Splat(splat::SplatOptions),
}

impl ViriBinpackOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        match self {
            Self::Interleave(options) => options.run(),
            Self::Head(options) => options.run(),
            Self::Count(options) => options.run(),
            Self::Splat(options) => options.run(),
        }
    }
}
