pub mod args;
pub mod emit;
pub mod fusion;
pub mod node;
pub mod op;
pub mod shape;

use std::{cell::RefCell, collections::HashSet, num::NonZeroUsize, sync::Arc, thread, time::Duration};

use args::GraphIRCompileArgs;
use emit::GraphIRStringFormat;
use fusion::FusionDescription;
use node::{AnnotatedNode, GraphIRNode, GraphIRNodeError};
use op::{GraphIROp, GraphIROpError};
use shape::Shape;

use crate::backend::{device::Device, tensor::Tensor};

use super::Graph;

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

    pub fn add_op(&mut self, operation: GraphIROp) -> Result<AnnotatedNode, GraphIRError> {
        let (shape, batched, requires_grad) = operation.output_info(self)?;
        self.add_node(None, Some(operation), shape, batched, requires_grad, None)
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

    pub fn compile<D: Device>(mut self, device: D, args: GraphIRCompileArgs) -> Result<Graph<D>, GraphIRError> {
        self.is_valid()?;

        if args.emit_ir {
            print!("{self}");
        }

        if args.allow_optimisations {
            self.optimise(&args)?;
            self.is_valid()?;
        }

        let root = self.root()?.idx;
        let root_data = self.get(root).unwrap();

        if !root_data.requires_grad || root_data.batched || root_data.shape != Shape::new(1, 1) {
            return Err(GraphIRError::Compilation(GraphIRCompileError::InvalidRootNode));
        }

        let device = Arc::new(device);

        let mut nodes = Vec::new();
        for node_data in &self.nodes {
            if let Some(GraphIRNode { shape, requires_grad, parent_operation, idx, sparse, .. }) = node_data.clone() {
                let tensor = Tensor::new(device.clone(), shape, requires_grad, parent_operation, sparse, idx);
                let tensor = tensor.map_err(|_| GraphIRError::Compilation(GraphIRCompileError::FailedToInitTensor));

                nodes.push(Some(RefCell::new(tensor?)));
            } else {
                nodes.push(None);
            }
        }

        let id_idx_pair = |&node| self.get(node).ok().map(|data| (data.id.clone().unwrap(), node));

        let inputs = self.inputs.iter().filter_map(id_idx_pair).collect();

        let weights = self.weights.iter().filter_map(id_idx_pair).collect();

        Ok(Graph { nodes, inputs, weights, device, profile: Default::default() })
    }

    pub fn optimise(&mut self, args: &GraphIRCompileArgs) -> Result<(), GraphIRError> {
        let mut optimistions = 0;

        if let Some(x) = args.fancy_ir_display {
            print!("\x1b[s{self}\x1b[u");
            thread::sleep(Duration::from_secs_f32(x));
        }

        while self.try_fusion_pass()? {
            optimistions += 1;

            if let Some(x) = args.fancy_ir_display {
                print!("\x1b[s{self}\x1b[u");
                thread::sleep(Duration::from_secs_f32(x));
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

    fn try_fusion_pass(&mut self) -> Result<bool, GraphIRError> {
        for node in (0..self.nodes.len()).rev() {
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
            let (shape, batched, requires_grad) = op.output_info(self)?;

            if shape != data.shape || data.batched != batched || data.requires_grad != requires_grad {
                return Err(GraphIRError::Node(GraphIRNodeError::NodeDataDoesNotMatchExpected));
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
