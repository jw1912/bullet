mod broadcast;
mod canonicalise;
mod constants;
mod decompose;
mod eliminate;
mod modify;

pub use broadcast::FoldBroadcasts;
pub use canonicalise::CanonicaliseInputs;
pub use constants::FoldConstants;
pub use decompose::DecomposeElementwise;
pub use eliminate::{EliminateCommonSubExpressions, EliminateCopies, EliminateUnusedOperations};
pub use modify::{AddOperation, RemoveOperation, ReplaceInput, SwapOutputs};

use crate::ir::{IR, IRTrace};

pub trait IrTransform: std::fmt::Debug + 'static {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace>;
}
