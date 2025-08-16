mod exchange;
mod fuse;

use std::rc::Rc;

use acyclib::graph::NodeId;
pub use exchange::*;
pub use fuse::*;

use crate::graph::ir::{operation::GraphIROperationCompilable, BackendMarker, GraphIR, GraphIRError};

pub trait GraphIRPass<B: BackendMarker> {
    fn try_pass(&self, ir: &mut GraphIR<B>) -> Result<bool, GraphIRError>;
}

pub trait GraphIRSimplePass<B: BackendMarker> {
    fn try_pass_on_node(&self, ir: &mut GraphIR<B>, node: NodeId) -> Result<bool, GraphIRError>;
}

impl<B: BackendMarker, T: GraphIRSimplePass<B>> GraphIRPass<B> for T {
    fn try_pass(&self, ir: &mut GraphIR<B>) -> Result<bool, GraphIRError> {
        for node in ir.topo_order()? {
            if self.try_pass_on_node(ir, node)? {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

#[allow(clippy::borrowed_box)]
fn downcast<B: BackendMarker, T: 'static>(op: &Rc<dyn GraphIROperationCompilable<B>>) -> Option<&T> {
    let op: &dyn std::any::Any = op.as_ref();
    op.downcast_ref()
}
