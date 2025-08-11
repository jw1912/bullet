mod exchange;
mod fuse;

pub use exchange::*;
pub use fuse::*;

use crate::graph::ir::{
    operation::GraphIROperationCompilable, transform::GraphIRTransform, BackendMarker, GraphIR, GraphIRError,
};

pub trait GraphIRPass<B: BackendMarker> {
    fn try_pass(&self, ir: &GraphIR<B>) -> Result<Option<GraphIRTransform<B>>, GraphIRError>;
}

pub trait GraphIRSimplePass<B: BackendMarker> {
    fn try_pass_on_node(&self, ir: &GraphIR<B>, node: usize) -> Result<Option<GraphIRTransform<B>>, GraphIRError>;
}

impl<B: BackendMarker, T: GraphIRSimplePass<B>> GraphIRPass<B> for T {
    fn try_pass(&self, ir: &GraphIR<B>) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        for node in ir.topo_order()? {
            if let Some(mut transform) = self.try_pass_on_node(ir, node)? {
                transform.delete.push(node);
                return Ok(Some(transform));
            }
        }

        Ok(None)
    }
}

#[allow(clippy::borrowed_box)]
pub fn downcast<B: BackendMarker, T: 'static>(op: &Option<Box<dyn GraphIROperationCompilable<B>>>) -> Option<&T> {
    op.as_ref().and_then(downcast_impl)
}

#[allow(clippy::borrowed_box)]
fn downcast_impl<B: BackendMarker, T: 'static>(op: &Box<dyn GraphIROperationCompilable<B>>) -> Option<&T> {
    let op: &dyn std::any::Any = op.as_ref();
    op.downcast_ref()
}
