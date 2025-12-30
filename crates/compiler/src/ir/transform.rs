pub mod canonicalise;
pub mod decompose;
pub mod destructive;
pub mod eliminate;
pub mod foldrules;
pub mod modify;
pub mod simplify;

pub use simplify::SimplifyPass;

use crate::ir::{IR, IRTrace};

pub trait IrTransform: std::fmt::Debug + 'static {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace>;
}
