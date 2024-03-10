pub mod inputs;
mod load;
pub mod outputs;

pub use load::{Feat, GpuDataLoader};

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    CReLU,
    SCReLU,
}
