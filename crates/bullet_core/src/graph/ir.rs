pub mod args;
pub mod emit;
pub mod fusion;
pub mod node;
pub mod op;

use std::{cell::RefCell, collections::HashSet, num::NonZeroUsize, sync::Arc};

use args::GraphIRCompileArgs;
use emit::GraphIRStringFormat;
use fusion::FusionDescription;
use node::AnnotatedNode;
use op::{GraphIROp, GraphIROpError};

use crate::backend::{
    device::{blas::Shape, Device},
    tensor::Tensor,
};

use super::Graph;

#[derive(Clone, Debug)]
pub struct GraphIRNode {
    pub id: Option<String>,
    pub size: usize,
    pub requires_grad: bool,
    pub parent_operation: Option<GraphIROp>,
    pub num_children: usize,
    pub own: AnnotatedNode,
}

impl GraphIRNode {
    pub fn is_valid(&self) -> Result<(), GraphIRNodeError> {
        if self.own.shape.size() != self.size {
            return Err(GraphIRNodeError::NodeDataDoesNotMatchExpected);
        }

        if let Some(op) = &self.parent_operation {
            let (shape, batched) = op.output_shape().unwrap();

            if shape != self.own.shape || self.own.can_be_batched != batched {
                return Err(GraphIRNodeError::NodeDataDoesNotMatchExpected);
            }
        }

        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub enum GraphIRNodeError {
    NodeWithIdAlreadyExists(String),
    NodeDataDoesNotMatchExpected,
    NodeDoesNotExist,
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
    MultipleRoots,
}

impl From<GraphIROpError> for GraphIRError {
    fn from(value: GraphIROpError) -> Self {
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

    pub fn root(&self) -> Result<AnnotatedNode, GraphIRError> {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");
        let root = self.roots.iter().next().ok_or(GraphIRError::MultipleRoots)?;

        Ok(self.get(*root)?.own)
    }

    pub fn compile<D: Device>(mut self, device: D, args: GraphIRCompileArgs) -> Result<Graph<D>, GraphIRError> {
        self.is_valid()?;

        if args.emit_ir {
            print!("{self}");
        }

        if args.allow_optimisations {
            self.optimise(&args)?;
        }

        self.is_valid()?;

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

        let id_idx_pair = |&node| self.get(node).ok().map(|data| (data.id.clone().unwrap(), node));

        let inputs = self.inputs.iter().filter_map(id_idx_pair).collect();

        let weights = self.weights.iter().filter_map(id_idx_pair).collect();

        Ok(Graph { nodes, root, inputs, weights, device, profile: Default::default() })
    }

    pub fn optimise(&mut self, args: &GraphIRCompileArgs) -> Result<(), GraphIRError> {
        let mut optimistions = 0;

        if let Some(x) = args.fancy_ir_display {
            print!("\x1b[s{self}\x1b[u");
            std::thread::sleep(std::time::Duration::from_secs_f32(x));
        }

        while self.optimisation_pass(fusion::fusion_pass)? {
            optimistions += 1;

            if let Some(x) = args.fancy_ir_display {
                print!("\x1b[s{self}\x1b[u");
                std::thread::sleep(std::time::Duration::from_secs_f32(x));
            }
        }

        if args.fancy_ir_display.is_some() {
            print!("{self}");
        }

        if args.emit_ir {
            println!("Optimisations: {optimistions}");
            let fmt = GraphIRStringFormat { fill_newlines: false, ..GraphIRStringFormat::default_colours() };
            print!("{}", self.to_formatted_string(&fmt).unwrap());
        }

        Ok(())
    }

    fn optimisation_pass<F: FnMut(&mut Self, usize) -> Result<bool, GraphIRError>>(
        &mut self,
        mut pass: F,
    ) -> Result<bool, GraphIRError> {
        for node in (0..self.nodes.len()).rev() {
            if self.get(node).is_ok() && pass(self, node)? {
                return Ok(true);
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

        new_nodes.sort_by_key(|x| x.own.idx);
        for new_data in new_nodes {
            self.replace_data(new_data.own.idx, new_data)?;
        }

        Ok(())
    }

    fn is_valid(&self) -> Result<(), GraphIRError> {
        for node in 0..self.nodes.len() {
            if let Ok(data) = self.get(node) {
                if data.own.idx != node {
                    return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
                }

                data.is_valid()?;

                if let Some(op) = &data.parent_operation {
                    for parent in op.nodes() {
                        let actual_parent =
                            self.nodes[parent.idx].as_ref().ok_or(GraphIRNodeError::NodeDoesNotExist)?;

                        let parent_node = actual_parent.own;

                        if parent.idx != parent_node.idx
                            || parent.can_be_batched != parent_node.can_be_batched
                            || parent.sparse != parent_node.sparse
                            || parent.shape.size() != parent_node.shape.size()
                        {
                            return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn delete_node(&mut self, node: usize) -> Result<(), GraphIRError> {
        if let Some(Some(op)) = self.nodes[node].as_ref().map(|x| x.parent_operation) {
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
