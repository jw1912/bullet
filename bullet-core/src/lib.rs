pub mod inputs;
mod load;
pub mod outputs;
mod rng;
pub mod util;

pub use load::{Feat, GpuDataLoader};
pub use rng::Rand;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    CReLU,
    SCReLU,
}
