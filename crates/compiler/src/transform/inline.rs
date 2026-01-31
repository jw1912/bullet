use crate::{IR, IRTrace, operation::SubGraph, transform::IRTransform};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineSubgraphs;

impl IRTransform for InlineSubgraphs {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        loop {
            let mut count = 0;

            for op in ir.operations() {
                if let Some(subgraph) = op.downcast::<SubGraph>() {
                    unimplemented!();
                }
            }

            if count == 0 {
                return Ok(());
            }
        }
    }
}
