pub mod args;
pub mod emit;
pub mod fusion;
pub mod node;
pub mod op;

use std::{cell::RefCell, collections::HashSet, num::NonZeroUsize, sync::Arc};

use args::GraphIRCompileArgs;
use node::AnnotatedNode;
use op::{GraphIROp, GraphIROpError};

use crate::backend::{
    device::{blas::Shape, Device},
    tensor::Tensor,
};

use super::Graph;

#[derive(Clone)]
pub struct GraphIRNode {
    pub id: Option<String>,
    pub size: usize,
    pub requires_grad: bool,
    pub parent_operation: Option<GraphIROp>,
    pub num_children: usize,
    pub own: AnnotatedNode,
}

#[derive(Debug, PartialEq)]
pub enum GraphIRNodeError {
    NodeWithIdAlreadyExists(String),
}

#[derive(Default)]
pub struct GraphIR {
    nodes: Vec<Option<GraphIRNode>>,
    roots: HashSet<usize>,
    inputs: HashSet<usize>,
    weights: HashSet<usize>,
    ids: HashSet<String>,
}

#[derive(Debug, PartialEq)]
pub enum GraphIRCompileError {
    MoreThanOneRoot,
    RootIsAnInput,
    RootIsAWeight,
    RootIsNonScalar,
    FailedToInitTensor,
}

#[derive(Debug, PartialEq)]
pub enum GraphIRError {
    Node(GraphIRNodeError),
    Op(GraphIROpError),
    Compilation(GraphIRCompileError),
}

impl From<GraphIROpError> for GraphIRError {
    fn from(value: GraphIROpError) -> Self {
        Self::Op(value)
    }
}

impl GraphIR {
    pub fn get(&self, idx: usize) -> Option<&GraphIRNode> {
        self.nodes[idx].as_ref()
    }

    pub fn add_node(
        &mut self,
        id: Option<String>,
        parent_operation: Option<GraphIROp>,
        shape: Shape,
        can_be_batched: bool,
        requires_grad: bool,
        sparse: Option<NonZeroUsize>,
    ) -> Result<AnnotatedNode, GraphIRError> {
        if let Some(id) = id.as_ref() {
            if self.ids.contains(id) {
                return Err(GraphIRError::Node(GraphIRNodeError::NodeWithIdAlreadyExists(id.clone())));
            }

            self.ids.insert(id.to_string());
        }

        let node = GraphIRNode {
            id,
            parent_operation,
            size: shape.size(),
            requires_grad,
            own: AnnotatedNode { idx: self.nodes.len(), shape, can_be_batched, sparse },
            num_children: 0,
        };

        if let Some(op) = &node.parent_operation {
            for parent in &op.nodes() {
                self.roots.remove(&parent.idx);
                if let Some(ir_node) = self.nodes[parent.idx].as_mut() {
                    ir_node.num_children += 1;
                }
            }
        }

        let own = node.own;
        self.nodes.push(Some(node));
        self.roots.insert(own.idx);

        Ok(own)
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

    pub fn add_op(&mut self, operation: GraphIROp, requires_grad: bool) -> Result<AnnotatedNode, GraphIRError> {
        let (shape, can_be_batched) = operation.output_shape()?;
        self.add_node(None, Some(operation), shape, can_be_batched, requires_grad, None)
    }

    pub fn root(&self) -> AnnotatedNode {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");
        self.get(*self.roots.iter().next().unwrap()).unwrap().own
    }

    pub fn compile<D: Device>(mut self, device: D, args: GraphIRCompileArgs) -> Result<Graph<D>, GraphIRError> {
        if args.emit_ir {
            print!("{self}");
        }

        self.optimise(&args);

        if self.roots.len() != 1 {
            return Err(GraphIRError::Compilation(GraphIRCompileError::MoreThanOneRoot));
        }

        let root = *self.roots.iter().next().unwrap();
        let root_data = self.get(root).unwrap();

        if !root_data.requires_grad {
            return Err(GraphIRError::Compilation(GraphIRCompileError::RootIsAnInput));
        }

        if self.weights.contains(&root) {
            return Err(GraphIRError::Compilation(GraphIRCompileError::RootIsAWeight));
        }

        if root_data.own.shape != Shape::new(1, 1) {
            return Err(GraphIRError::Compilation(GraphIRCompileError::RootIsNonScalar));
        }

        let device = Arc::new(device);

        let mut nodes = Vec::new();
        for node_data in &self.nodes {
            if let Some(GraphIRNode { size, requires_grad, parent_operation, own, .. }) = node_data.clone() {
                let tensor = Tensor::new(device.clone(), size, requires_grad, parent_operation, own);
                let tensor = tensor.map_err(|_| GraphIRError::Compilation(GraphIRCompileError::FailedToInitTensor));

                nodes.push(Some(RefCell::new(tensor?)));
            } else {
                nodes.push(None);
            }
        }

        let id_idx_pair = |&node| self.get(node).map(|data| (data.id.clone().unwrap(), node));

        let inputs = self.inputs.iter().filter_map(id_idx_pair).collect();

        let weights = self.weights.iter().filter_map(id_idx_pair).collect();

        Ok(Graph { nodes, root, inputs, weights, device, profile: Default::default() })
    }

    pub fn optimise(&mut self, args: &GraphIRCompileArgs) {
        let mut fusions = 0;

        if args.allow_fusion {
            while self.optimisation_pass(fusion::fusion_pass) {
                fusions += 1;
            }

            if args.emit_ir {
                println!("Fusions: {fusions}");
                print!("{self}");
            }
        }
    }

    fn optimisation_pass<F: FnMut(&mut Self, usize) -> bool>(&mut self, mut pass: F) -> bool {
        for node in (0..self.nodes.len()).rev() {
            if let Some(data) = self.get(node) {
                assert_eq!(data.own.idx, node);

                if pass(self, node) {
                    return true;
                }
            }
        }

        false
    }
}
