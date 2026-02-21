mod count;
mod head;
mod interleave;

use structopt::StructOpt;

#[derive(StructOpt)]
pub enum MontyBinpackOptions {
    Head(head::HeadOptions),
    Interleave(interleave::InterleaveOptions),
    Count(count::CountOptions),
}

impl MontyBinpackOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        match self {
            Self::Interleave(options) => options.run(),
            Self::Head(options) => options.run(),
            Self::Count(options) => options.run(),
        }
    }
}
