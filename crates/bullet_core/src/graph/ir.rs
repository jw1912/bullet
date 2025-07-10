pub mod fusion;
pub mod node;
pub mod operation;
pub mod shape;

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    sync::Arc,
};

use fusion::FusionDescription;
use node::{AnnotatedNode, GraphIRNode, GraphIRNodeError};
use operation::*;
use shape::Shape;

use crate::{
    backend::{device::Device, tensor::Tensor},
    graph::{instruction::Set, Graph, GraphFunction, NodeId, NodeIdTy},
};

#[derive(Default)]
pub struct GraphIR {
    nodes: Vec<Option<GraphIRNode>>,
    inputs: HashSet<usize>,
    weights: HashSet<usize>,
    ids: HashSet<String>,
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

impl GraphIR {
    pub fn get(&self, idx: usize) -> Result<&GraphIRNode, GraphIRError> {
        self.nodes[idx].as_ref().ok_or(GraphIRError::Node(GraphIRNodeError::NodeDoesNotExist))
    }

    pub fn get_mut(&mut self, idx: usize) -> Result<&mut GraphIRNode, GraphIRError> {
        self.nodes[idx].as_mut().ok_or(GraphIRError::Node(GraphIRNodeError::NodeDoesNotExist))
    }

    pub fn add_node(
        &mut self,
        id: Option<String>,
        parent_operation: Option<Box<dyn GraphIROperation>>,
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

        let idx = self.nodes.len();
        let annotated = AnnotatedNode { idx, shape };

        let node = GraphIRNode { id, parent_operation, shape, requires_grad, idx, num_children: 0, batched, sparse };

        if let Some(op) = &node.parent_operation {
            for parent in &op.nodes() {
                if let Some(ir_node) = self.nodes[parent.idx].as_mut() {
                    ir_node.num_children += 1;
                }
            }
        }

        self.nodes.push(Some(node));

        Ok(annotated)
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

    pub fn add_op(&mut self, operation: impl GraphIROperation) -> Result<AnnotatedNode, GraphIRError> {
        let shape = operation.output_shape(self)?;
        let batched = operation.output_batched(self)?;
        let requires_grad = operation.output_requires_grad(self)?;
        self.add_node(None, Some(Box::new(operation)), shape, batched, requires_grad, None)
    }

    pub fn root(&self) -> Result<AnnotatedNode, GraphIRError> {
        let roots =
            self.nodes.iter().filter(|node| node.as_ref().map(|x| x.num_children == 0).unwrap_or(false)).count();

        if roots != 1 {
            return Err(GraphIRError::MultipleRoots);
        }

        let idx = self.nodes.len() - 1;
        let data = self.get(idx)?;

        Ok(AnnotatedNode { idx, shape: data.shape })
    }

    pub fn compile<D: Device>(mut self, device: D) -> Result<Graph<D>, GraphIRError> {
        self.is_valid()?;
        self.optimise()?;
        self.is_valid()?;

        let root = self.root()?.idx;
        let root_data = self.get(root).unwrap();

        if !root_data.requires_grad || root_data.batched || root_data.shape != Shape::new(1, 1) {
            return Err(GraphIRError::Compilation(GraphIRCompileError::InvalidRootNode));
        }

        let device = Arc::new(device);

        let mut nodes = HashMap::new();
        let mut forward = GraphFunction::default();
        let mut backward = GraphFunction::default();
        let mut zero_grads = GraphFunction::default();

        let id_idx_pair = |&node| self.get(node).ok().map(|data| (data.id.clone().unwrap(), node));

        let inputs = self.inputs.iter().filter_map(id_idx_pair).collect();

        let weights = self.weights.iter().filter_map(id_idx_pair).collect();

        for GraphIRNode { idx, shape, sparse, requires_grad, parent_operation, .. } in self.nodes.into_iter().flatten()
        {
            let values = Tensor::new(device.clone(), shape, sparse)
                .map_err(|_| GraphIRError::Compilation(GraphIRCompileError::FailedToInitTensor))?;

            nodes.insert(NodeId::new(idx, NodeIdTy::Values), RefCell::new(values));

            if requires_grad {
                let grads = Tensor::new(device.clone(), shape, sparse)
                    .map_err(|_| GraphIRError::Compilation(GraphIRCompileError::FailedToInitTensor))?;

                let id = NodeId::new(idx, NodeIdTy::Gradients);
                nodes.insert(id, RefCell::new(grads));

                zero_grads.push(Set(id, 0.0));
            }

            if let Some(op) = parent_operation {
                let opname = format!("{op:?}");

                if let Some(compilable) = downcast_op::<D>(op) {
                    forward.extend(compilable.forward_pass());

                    if requires_grad {
                        let mut this_bwd = compilable.backward_pass();
                        this_bwd.extend(backward);
                        backward = this_bwd;
                    }
                } else {
                    return Err(GraphIRError::Compilation(GraphIRCompileError::UnsupportedOperation(opname)));
                }
            }
        }

        let functions = [("forward", forward), ("backward", backward), ("zero_grads", zero_grads)]
            .into_iter()
            .map(|(x, y)| (x.to_string(), y))
            .collect();

        Ok(Graph { nodes, inputs, weights, functions, device })
    }

    fn optimise(&mut self) -> Result<(), GraphIRError> {
        while self.try_fusion_pass()? {}

        Ok(())
    }

    fn try_fusion_pass(&mut self) -> Result<bool, GraphIRError> {
        for node in 0..self.nodes.len() {
            if self.get(node).is_ok() {
                if let Some(mut desc) = fusion::search_for_fusion(self, node)? {
                    desc.eliminated.push(node);
                    self.apply_fusion(desc)?;
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn apply_fusion(&mut self, desc: FusionDescription) -> Result<(), GraphIRError> {
        let FusionDescription { mut eliminated, mut new_nodes } = desc;

        eliminated.sort();
        for dead in eliminated.into_iter().rev() {
            self.delete_node(dead)?;
        }

        new_nodes.sort_by_key(|x| x.idx);
        for new_data in new_nodes {
            self.replace_data(new_data.idx, new_data)?;
        }

        Ok(())
    }

    fn is_valid(&self) -> Result<(), GraphIRError> {
        for node in 0..self.nodes.len() {
            if let Ok(data) = self.get(node) {
                if data.idx != node {
                    return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
                }

                self.is_data_valid(data)?;

                if let Some(op) = &data.parent_operation {
                    for parent in op.nodes() {
                        let actual_parent =
                            self.nodes[parent.idx].as_ref().ok_or(GraphIRNodeError::NodeDoesNotExist)?;

                        if parent.idx != actual_parent.idx || parent.shape.size() != actual_parent.shape.size() {
                            return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn is_data_valid(&self, data: &GraphIRNode) -> Result<(), GraphIRError> {
        if let Some(op) = &data.parent_operation {
            let shape = op.output_shape(self)?;
            let batched = op.output_batched(self)?;
            let requires_grad = op.output_requires_grad(self)?;

            if shape != data.shape || data.batched != batched || data.requires_grad != requires_grad {
                return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
            }
        }

        Ok(())
    }

    fn delete_node(&mut self, node: usize) -> Result<(), GraphIRError> {
        if let Some(Some(op)) = self.nodes[node].as_ref().map(|x| &x.parent_operation) {
            for parent in op.nodes() {
                self.get_mut(parent.idx)?.num_children -= 1;
            }
        }

        self.nodes[node] = None;

        Ok(())
    }

    fn replace_data(&mut self, node: usize, data: GraphIRNode) -> Result<(), GraphIRError> {
        if let Some(op) = data.parent_operation.as_ref() {
            for parent in op.nodes() {
                self.get_mut(parent.idx)?.num_children += 1;
            }
        }

        self.nodes[node] = Some(data);

        Ok(())
    }
}

fn downcast_op<D: Device>(x: Box<dyn GraphIROperation>) -> Option<Box<dyn GraphIROperationCompile<D>>> {
    let x: Box<dyn std::any::Any> = Box::new(x);
    x.downcast::<Box<dyn GraphIROperationCompile<D>>>().map(|x| *x).ok()
}
