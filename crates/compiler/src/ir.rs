mod analysis;
mod display;
mod evaluate;
pub mod node;
pub mod operation;
mod transform;

use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use node::{IrNode, IrNodeId, IrType};
use operation::{IrOperation, IrOperationId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrError(pub String);

impl<T: Into<String>> From<T> for IrError {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

#[derive(Clone, Default, Debug)]
pub struct IrGraph {
    nodes: HashMap<IrNodeId, IrNode>,
    ops: HashMap<IrOperationId, IrOperation>,
    links: HashMap<IrNodeId, IrOperationId>,
    outputs: HashSet<IrNodeId>,
}

impl IrGraph {
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_parent_op(&self, node: IrNodeId) -> Result<IrOperationId, IrError> {
        self.links.get(&node).cloned().ok_or(format!("IrGraph::get_parent_op: node {node:?} does not exist!").into())
    }

    pub fn get_op(&self, op: IrOperationId) -> Result<&IrOperation, IrError> {
        self.ops.get(&op).ok_or(format!("IrGraph::get_op: operation {op:?} does not exist!").into())
    }

    pub fn get_op_mut(&mut self, op: IrOperationId) -> Result<&mut IrOperation, IrError> {
        self.ops.get_mut(&op).ok_or(format!("IrGraph::get_op_mut: operation {op:?} does not exist!").into())
    }

    pub fn get_node(&self, node: IrNodeId) -> Result<&IrNode, IrError> {
        self.nodes.get(&node).ok_or(format!("IrGraph::get_node: node {node:?} does not exist!").into())
    }

    pub fn get_node_mut(&mut self, node: IrNodeId) -> Result<&mut IrNode, IrError> {
        self.nodes.get_mut(&node).ok_or(format!("IrGraph::get_node_mut: node {node:?} does not exist!").into())
    }

    pub fn get_node_type(&self, node: IrNodeId) -> Result<IrType, IrError> {
        self.get_node(node).map(IrNode::ty)
    }

    pub fn is_output(&self, node: IrNodeId) -> bool {
        self.outputs.contains(&node)
    }

    pub fn register_output(&mut self, node: IrNodeId) {
        self.outputs.insert(node);
    }

    pub fn unregister_output(&mut self, node: IrNodeId) {
        self.outputs.remove(&node);
    }

    pub fn optimise(&mut self) -> Result<(), IrError> {
        self.decompose_elementwise()?;
        self.canonicalise()?;
        self.eliminate_unused_ops()?;
        self.fold_constants()?;
        self.eliminate_common_subexprs()
    }
}
