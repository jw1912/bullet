mod exchange;
mod fuse;

use std::{fmt::Debug, rc::Rc};

use acyclib::graph::NodeId;
pub use exchange::*;
pub use fuse::*;

use crate::graph::ir::{BackendMarker, GraphIR, GraphIRError, operation::GraphIROperationCompilable};

pub trait GraphIRPass<B: BackendMarker>: Debug {
    fn try_pass(&self, ir: &mut GraphIR<B>) -> Result<bool, GraphIRError>;
}

pub trait GraphIRSimplePass<B: BackendMarker>: Debug {
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

pub fn downcast<B: BackendMarker, T: Clone + 'static>(op: &Rc<dyn GraphIROperationCompilable<B>>) -> Option<T> {
    let op: &dyn std::any::Any = op.as_ref();
    op.downcast_ref().cloned()
}
