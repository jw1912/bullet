pub mod compile;
pub mod node;
pub mod operation;
pub mod passes;
pub mod properties;
pub mod shape;
pub mod transform;

use std::{
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

use node::{AnnotatedNode, GraphIRNode, NodeInfo};
use operation::*;
use shape::Shape;

use crate::device::Device;

pub trait BackendMarker: Copy + Default + 'static {
    type Backend: Device;
}

#[derive(Debug, Default)]
pub struct GraphIRCompileOptions {
    pub dump_graphviz: Option<String>,
}

#[derive(Default)]
pub struct GraphIR<B: BackendMarker> {
    nodes: HashMap<usize, GraphIRNode<B>>,
    counter: AtomicUsize,
    inputs: HashSet<usize>,
    weights: HashSet<usize>,
    ids: HashSet<String>,
    opts: GraphIRCompileOptions,
}

pub struct GraphIRNodeInfo {
    nodes: HashMap<usize, NodeInfo>,
}

impl GraphIRNodeInfo {
    pub fn get(&self, node: usize) -> Option<NodeInfo> {
        self.nodes.get(&node).copied()
    }
}

#[derive(Debug)]
pub enum GraphIRError {
    Op(GraphIROperationError),
    MultipleRoots,
    CannotBeTopologicallyOrdered,
    NodeAlreadyExists,
    NodeWithIdAlreadyExists(String),
    NodeDataDoesNotMatchExpected,
    NodeDoesNotExist,
    NodeHasInvalidNumberOfChildren,
}

impl From<GraphIROperationError> for GraphIRError {
    fn from(value: GraphIROperationError) -> Self {
        Self::Op(value)
    }
}

impl<B: BackendMarker> GraphIR<B> {
    pub fn get(&self, idx: usize) -> Result<&GraphIRNode<B>, GraphIRError> {
        self.nodes.get(&idx).ok_or(GraphIRError::NodeDoesNotExist)
    }

    pub fn get_mut(&mut self, idx: usize) -> Result<&mut GraphIRNode<B>, GraphIRError> {
        self.nodes.get_mut(&idx).ok_or(GraphIRError::NodeDoesNotExist)
    }

    pub fn new_idx(&self) -> usize {
        self.counter.fetch_add(1, Ordering::SeqCst)
    }

    pub fn add_leaf(
        &mut self,
        id: Option<String>,
        shape: Shape,
        batched: bool,
        requires_grad: bool,
        sparse: Option<NonZeroUsize>,
    ) -> Result<AnnotatedNode, GraphIRError> {
        let idx = self.new_idx();
        let annotated = AnnotatedNode { idx, shape };

        self.insert_node(GraphIRNode {
            id,
            parent_operation: None,
            info: NodeInfo { shape, requires_grad, batched, sparse },
            idx,
            num_children: 0,
        })?;

        Ok(annotated)
    }

    pub fn add_constant(&mut self, shape: Shape) -> Result<AnnotatedNode, GraphIRError> {
        let node = self.add_leaf(None, shape, false, false, None)?;
        Ok(node)
    }

    pub fn add_dense_input(&mut self, id: &str, shape: Shape) -> Result<AnnotatedNode, GraphIRError> {
        let node = self.add_leaf(Some(id.to_string()), shape, true, false, None)?;
        self.inputs.insert(node.idx);
        Ok(node)
    }

    pub fn add_sparse_input(&mut self, id: &str, shape: Shape, nnz: usize) -> Result<AnnotatedNode, GraphIRError> {
        let nnz = NonZeroUsize::try_from(nnz).unwrap();
        let node = self.add_leaf(Some(id.to_string()), shape, true, false, Some(nnz))?;
        self.inputs.insert(node.idx);
        Ok(node)
    }

    pub fn add_unbatched_input(
        &mut self,
        id: &str,
        shape: Shape,
        sparse: Option<usize>,
    ) -> Result<AnnotatedNode, GraphIRError> {
        let sparse = sparse.map(|nnz| NonZeroUsize::try_from(nnz).unwrap());
        let node = self.add_leaf(Some(id.to_string()), shape, false, false, sparse)?;
        self.inputs.insert(node.idx);
        Ok(node)
    }

    pub fn add_weights(&mut self, id: &str, shape: Shape) -> Result<AnnotatedNode, GraphIRError> {
        let node = self.add_leaf(Some(id.to_string()), shape, false, true, None)?;
        self.weights.insert(node.idx);
        Ok(node)
    }

    pub fn add_op(&mut self, operation: impl GraphIROperationCompilable<B>) -> Result<AnnotatedNode, GraphIRError> {
        let node = self.result_of(operation)?;
        let annotated = AnnotatedNode { idx: node.idx, shape: node.info.shape };
        self.insert_node(node)?;
        Ok(annotated)
    }

    pub fn set_compile_opts(&mut self, opts: GraphIRCompileOptions) {
        self.opts = opts;
    }

    pub fn as_graphviz(&self, prefix: &str) -> Result<String, std::fmt::Error> {
        use std::fmt::Write;

        let mut s = String::new();

        for node in self.topo_order().unwrap() {
            if let Ok(data) = self.get(node) {
                if let Some(op) = &data.parent_operation {
                    let opname = op.shorthand();
                    writeln!(&mut s, "{prefix}{node} [label=\"{opname}\"];")?;

                    for parent in op.nodes() {
                        writeln!(&mut s, "{prefix}{} -> {prefix}{node};", parent.idx)?;
                    }
                } else {
                    let opname = data.id.clone().unwrap_or("__constant".to_string());
                    writeln!(&mut s, "{prefix}{node} [label=\"{opname}\", style=filled, color=lightblue];")?;
                }
            }
        }

        Ok(s)
    }
}
