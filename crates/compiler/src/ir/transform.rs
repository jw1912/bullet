mod broadcast;
mod canonicalise;
mod decompose;
pub mod eliminate;
pub mod fold;
pub(crate) mod modify;

pub use broadcast::FoldBroadcasts;
pub use canonicalise::CanonicaliseCommutativeInputs;
pub use decompose::DecomposeElementwise;
pub use fold::FoldPass;

use crate::ir::{IR, IRTrace};

pub trait IrTransform: std::fmt::Debug + 'static {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace>;
}
