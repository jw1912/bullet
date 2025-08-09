pub mod compile;
pub mod node;
pub mod operation;
pub mod passes;
pub mod shape;
pub mod transform;

use std::{
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

use node::{AnnotatedNode, GraphIRNode, GraphIRNodeError, NodeInfo};
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
    leafs: HashSet<usize>,
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

#[derive(Debug, PartialEq)]
pub enum GraphIRCompileError {
    InvalidRootNode,
    FailedToInitTensor,
    UnsupportedOperation(String),
}

#[derive(Debug)]
pub enum GraphIRError {
    Node(GraphIRNodeError),
    Op(GraphIROperationError),
    Compilation(GraphIRCompileError),
    MultipleRoots,
    NodeAlreadyExists,
    NodeDoesNotExist,
    CannotBeTopologicallyOrdered,
}

impl From<GraphIROperationError> for GraphIRError {
    fn from(value: GraphIROperationError) -> Self {
        Self::Op(value)
    }
}

impl From<GraphIRNodeError> for GraphIRError {
    fn from(value: GraphIRNodeError) -> Self {
        Self::Node(value)
    }
}

impl<B: BackendMarker> GraphIR<B> {
    pub fn get(&self, idx: usize) -> Result<&GraphIRNode<B>, GraphIRError> {
        self.nodes.get(&idx).ok_or(GraphIRError::Node(GraphIRNodeError::NodeDoesNotExist))
    }

    pub fn get_mut(&mut self, idx: usize) -> Result<&mut GraphIRNode<B>, GraphIRError> {
        self.nodes.get_mut(&idx).ok_or(GraphIRError::Node(GraphIRNodeError::NodeDoesNotExist))
    }

    pub fn new_idx(&self) -> usize {
        self.counter.fetch_add(1, Ordering::SeqCst)
    }

    pub fn add_node(
        &mut self,
        id: Option<String>,
        parent_operation: Option<Box<dyn GraphIROperationCompilable<B>>>,
        shape: Shape,
        batched: bool,
        requires_grad: bool,
        sparse: Option<NonZeroUsize>,
    ) -> Result<AnnotatedNode, GraphIRError> {
        if let Some(id) = id.as_ref() {
            if self.ids.contains(id) {
                return Err(GraphIRError::Node(GraphIRNodeError::NodeWithIdAlreadyExists(id.clone())));
            }

            self.ids.insert(id.to_string());
        }

        let idx = self.new_idx();
        let annotated = AnnotatedNode { idx, shape };

        self.insert_node(GraphIRNode {
            id,
            parent_operation,
            info: NodeInfo { shape, requires_grad, batched, sparse },
            idx,
            num_children: 0,
        })?;

        Ok(annotated)
    }

    pub fn add_constant(&mut self, shape: Shape) -> Result<AnnotatedNode, GraphIRError> {
        let node = self.add_node(None, None, shape, false, false, None)?;
        Ok(node)
    }

    pub fn add_dense_input(&mut self, id: &str, shape: Shape) -> Result<AnnotatedNode, GraphIRError> {
        let node = self.add_node(Some(id.to_string()), None, shape, true, false, None)?;
        self.inputs.insert(node.idx);
        Ok(node)
    }

    pub fn add_sparse_input(&mut self, id: &str, shape: Shape, nnz: usize) -> Result<AnnotatedNode, GraphIRError> {
        let nnz = NonZeroUsize::try_from(nnz).unwrap();
        let node = self.add_node(Some(id.to_string()), None, shape, true, false, Some(nnz))?;
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
        let node = self.add_node(Some(id.to_string()), None, shape, false, false, sparse)?;
        self.inputs.insert(node.idx);
        Ok(node)
    }

    pub fn add_weights(&mut self, id: &str, shape: Shape) -> Result<AnnotatedNode, GraphIRError> {
        let node = self.add_node(Some(id.to_string()), None, shape, false, true, None)?;
        self.weights.insert(node.idx);
        Ok(node)
    }

    pub fn topo_order(&self) -> Result<Vec<usize>, GraphIRError> {
        let mut edges: HashMap<usize, HashSet<usize>> = self.nodes.keys().map(|idx| (*idx, HashSet::new())).collect();
        let mut edgest: HashMap<usize, HashSet<usize>> = self.nodes.keys().map(|idx| (*idx, HashSet::new())).collect();

        for (&idx, data) in self.nodes.iter() {
            assert_eq!(idx, data.idx);

            if let Some(op) = &data.parent_operation {
                for node in op.nodes() {
                    edges.get_mut(&node.idx).unwrap().insert(idx);
                    edgest.get_mut(&idx).unwrap().insert(node.idx);
                }
            }
        }

        let mut leafs: HashSet<usize> = self.leafs.clone();

        let mut topo = Vec::new();

        loop {
            if leafs.is_empty() {
                break;
            }

            let n = *leafs.iter().next().unwrap();
            leafs.remove(&n);
            topo.push(n);

            let children = edges.get(&n).unwrap().clone();
            for child in children {
                edges.get_mut(&n).unwrap().remove(&child);

                let parents = edgest.get_mut(&child).unwrap();
                parents.remove(&n);
                if parents.is_empty() {
                    leafs.insert(child);
                }
            }
        }

        if edges.values().all(HashSet::is_empty) && edgest.values().all(HashSet::is_empty) {
            Ok(topo)
        } else {
            Err(GraphIRError::CannotBeTopologicallyOrdered)
        }
    }

    pub fn add_op(&mut self, operation: impl GraphIROperationCompilable<B>) -> Result<AnnotatedNode, GraphIRError> {
        let shape = operation.output_shape(self)?;
        let batched = operation.output_batched(self)?;
        let requires_grad = operation.output_requires_grad(self)?;
        self.add_node(None, Some(Box::new(operation)), shape, batched, requires_grad, None)
    }

    pub fn root(&self) -> Result<AnnotatedNode, GraphIRError> {
        let roots = self.nodes.values().filter(|node| node.num_children == 0).count();

        if roots != 1 {
            return Err(GraphIRError::MultipleRoots);
        }

        let idx = *self.topo_order()?.last().unwrap();
        let data = self.get(idx)?;

        Ok(AnnotatedNode { idx, shape: data.info.shape })
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
                    write!(&mut s, "{prefix}{node} [label=\"{opname}\"];")?;

                    for parent in op.nodes() {
                        write!(&mut s, "{prefix}{} -> {prefix}{node};", parent.idx)?;
                    }
                } else {
                    let opname = data.id.clone().unwrap_or("__constant".to_string());
                    write!(&mut s, "{prefix}{node} [label=\"{opname}\", style=filled, color=lightblue];")?;
                }
            }
        }

        Ok(s)
    }

    fn optimise(&mut self) -> Result<(), GraphIRError> {
        while self.try_fusion_pass()? {}

        Ok(())
    }

    fn try_fusion_pass(&mut self) -> Result<bool, GraphIRError> {
        for node in self.topo_order()? {
            if self.get(node).is_ok() {
                if let Some(mut transform) = passes::search_for_fusion(self, node)? {
                    transform.eliminated.push(node);
                    self.apply_transform(transform)?;
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn is_valid(&self) -> Result<(), GraphIRError> {
        let mut children_count: HashMap<_, _> = self.nodes.keys().map(|&idx| (idx, 0)).collect();

        for node in self.topo_order()? {
            if let Ok(data) = self.get(node) {
                if data.idx != node {
                    return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
                }

                self.is_data_valid(data)?;

                if let Some(op) = &data.parent_operation {
                    for parent in op.nodes() {
                        *children_count.get_mut(&parent.idx).unwrap() += 1;

                        let actual_parent = self.nodes.get(&parent.idx).ok_or(GraphIRNodeError::NodeDoesNotExist)?;

                        if parent.idx != actual_parent.idx || parent.shape.size() != actual_parent.info.shape.size() {
                            return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
                        }
                    }
                }
            }
        }

        for idx in self.nodes.keys() {
            if children_count.get(idx).copied() != self.nodes.get(idx).map(|x| x.num_children) {
                return Err(GraphIRError::Node(GraphIRNodeError::InvalidNumberOfChildren));
            }
        }

        Ok(())
    }

    fn is_data_valid(&self, data: &GraphIRNode<B>) -> Result<(), GraphIRError> {
        if let Some(op) = &data.parent_operation {
            let shape = op.output_shape(self)?;
            let batched = op.output_batched(self)?;
            let requires_grad = op.output_requires_grad(self)?;

            if data.info.shape != shape || data.info.batched != batched || data.info.requires_grad != requires_grad {
                return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
            }
        }

        Ok(())
    }
}
