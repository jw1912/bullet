pub(crate) mod builder;

use crate::default::Trainer;

pub use builder::{NoOutputBuckets, ValueTrainerBuilder};

pub use crate::default::loader;

/// For now `ValueTrainer` just aliases the existing `Trainer`,
/// because the only improvements for now are in the **construction**
/// of the trainer via `ValueTrainerBuilder`.
pub type ValueTrainer<Opt, Inp, Out> = Trainer<Opt, Inp, Out>;
