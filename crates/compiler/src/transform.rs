pub mod canonicalise;
pub mod eliminate;
pub mod foldrules;
pub mod inline;
pub mod modify;
pub mod ordering;
pub mod rewriterules;

pub use canonicalise::CanonicalisePass;

use crate::{IR, IRTrace};

pub trait IRTransform: std::fmt::Debug + 'static {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace>;
}
