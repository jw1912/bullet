pub mod canonicalise;
pub mod decompose;
pub mod eliminate;
pub mod foldrules;
pub mod modify;
pub mod ordering;
pub mod rewriterules;

pub use canonicalise::CanonicalisePass;

use crate::ir::{IR, IRTrace};

pub trait IrTransform: std::fmt::Debug + 'static {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace>;
}
