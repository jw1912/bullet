use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
    num::NonZeroUsize,
    sync::Arc,
};

use super::{
    error::GraphError,
    operation::{GraphBuilderError, GraphBuilderErrorType, Operation},
    Graph,
};
use crate::backend::{shape::Shape, tensor::Tensor, Device, OperationError};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Node {
    pub idx: usize,
    pub(crate) shape: Shape,
    pub(crate) sparse: Option<NonZeroUsize>,
    pub(crate) can_be_batched: bool,
}

impl Node {
    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn reshape(mut self, shape: Shape) -> Result<Self, GraphBuilderErrorType> {
        if self.shape.size() == shape.size() {
            self.shape = shape;
            Ok(self)
        } else {
            Err(GraphBuilderErrorType::MismatchedInputShapes(vec![self.shape, shape]))
        }
    }

    pub fn is_sparse(&self) -> bool {
        self.sparse.is_some()
    }
}

pub(crate) struct NodeData {
    id: Option<String>,
    size: usize,
    requires_grad: bool,
    parent_operation: Option<Operation>,
    own: Node,
}

impl NodeData {
    fn new(
        id: Option<String>,
        parent_operation: Option<Operation>,
        size: usize,
        can_be_batched: bool,
        requires_grad: bool,
        sparse: Option<NonZeroUsize>,
    ) -> Self {
        let own = Node { idx: usize::MAX, shape: Shape::new(usize::MAX, usize::MAX), can_be_batched, sparse };
        Self { id, size, requires_grad, parent_operation, own }
    }
}

#[derive(Default)]
pub struct GraphBuilder {
    nodes: Vec<Option<NodeData>>,
    roots: HashSet<usize>,
    inputs: HashSet<usize>,
    weights: HashSet<usize>,
    ids: HashSet<String>,
}

impl GraphBuilder {
    pub(crate) fn get(&self, idx: usize) -> Option<&NodeData> {
        self.nodes[idx].as_ref()
    }

    fn create_node(
        &mut self,
        mut data: NodeData,
        shape: Shape,
        sparse: Option<NonZeroUsize>,
    ) -> Result<Node, GraphBuilderErrorType> {
        if let Some(id) = data.id.as_ref() {
            if self.ids.contains(id) {
                return Err(GraphBuilderErrorType::NodeWithIdAlreadyExists);
            }

            self.ids.insert(id.to_string());
        }

        let node = Node { idx: self.nodes.len(), shape, can_be_batched: data.own.can_be_batched, sparse };
        data.own = node;

        if let Some(op) = &data.parent_operation {
            for parent in &op.nodes() {
                self.roots.remove(&parent.idx);
            }
        }

        self.nodes.push(Some(data));
        self.roots.insert(node.idx);

        Ok(node)
    }

    pub fn create_dense_input(&mut self, id: &str, shape: Shape) -> Result<Node, GraphBuilderErrorType> {
        let data = NodeData::new(Some(id.to_string()), None, shape.size(), true, false, None);
        let node = self.create_node(data, shape, None)?;

        self.inputs.insert(node.idx);

        Ok(node)
    }

    pub fn create_sparse_input(&mut self, id: &str, shape: Shape, nnz: usize) -> Result<Node, GraphBuilderErrorType> {
        let data = NodeData::new(Some(id.to_string()), None, shape.size(), true, false, None);
        let node = self.create_node(data, shape, Some(NonZeroUsize::try_from(nnz).unwrap()))?;

        self.inputs.insert(node.idx);

        Ok(node)
    }

    pub fn create_unbatched_input(
        &mut self,
        id: &str,
        shape: Shape,
        sparse: Option<usize>,
    ) -> Result<Node, GraphBuilderErrorType> {
        let sparse = sparse.map(|nnz| NonZeroUsize::try_from(nnz).unwrap());
        let data = NodeData::new(Some(id.to_string()), None, shape.size(), false, false, sparse);
        let node = self.create_node(data, shape, sparse)?;

        self.inputs.insert(node.idx);

        Ok(node)
    }

    pub fn create_weights(&mut self, id: &str, shape: Shape) -> Result<Node, GraphBuilderErrorType> {
        let data = NodeData::new(Some(id.to_string()), None, shape.size(), false, true, None);
        let node = self.create_node(data, shape, None)?;

        self.weights.insert(node.idx);

        Ok(node)
    }

    pub fn create_result_of_operation(
        &mut self,
        operation: Operation,
        requires_grad: bool,
    ) -> Result<Node, GraphBuilderError> {
        match operation.output_shape() {
            Ok(shape) => {
                let data = NodeData::new(None, Some(operation), shape.size(), true, requires_grad, None);
                self.create_node(data, shape, None).map_err(|e| GraphBuilderError::new(&operation, e))
            }
            Err(s) => Err(s),
        }
    }

    pub fn root(&self) -> Node {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");
        self.get(*self.roots.iter().next().unwrap()).unwrap().own
    }

    fn try_fusion_pass(&mut self) -> bool {
        for node in (0..self.nodes.len()).rev() {
            if let Some(data) = self.get(node) {
                let own = data.own.idx;

                if let Some((eliminated, new_data)) = self.search_for_fusion(own) {
                    for dead in eliminated {
                        self.nodes[dead] = None;
                    }

                    self.nodes[node] = Some(new_data);

                    return true;
                }
            }
        }

        false
    }

    pub fn optimise(&mut self) {
        while self.try_fusion_pass() {}
    }

    pub fn build<D: Device>(self, device: D) -> Result<Graph<D>, GraphError<D::DeviceError>> {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");

        let root = *self.roots.iter().next().unwrap();
        let root_data = self.get(root).unwrap();

        assert!(root_data.requires_grad, "Output cannot be an input!");
        assert!(!self.weights.contains(&root), "Can't output trainable weights!");
        assert_eq!(root_data.own.shape, Shape::new(1, 1), "Graph output must be scalar!");
        assert_eq!(root_data.size, 1);

        let device = Arc::new(device);

        let mut nodes = Vec::new();
        for node_data in &self.nodes {
            if let Some(data) = node_data {
                let tensor =
                    Tensor::new(device.clone(), data.size, data.requires_grad, data.parent_operation, data.own);
                let tensor = tensor.map_err(OperationError::from);

                nodes.push(Some(RefCell::new(tensor?)));
            } else {
                nodes.push(None);
            }
        }

        let id_idx_pair = |&node| self.get(node).map(|data| (data.id.clone().unwrap(), node));

        let inputs = self.inputs.iter().filter_map(id_idx_pair).collect::<HashMap<_, _>>();

        let weights = self.weights.iter().filter_map(id_idx_pair).collect::<HashMap<_, _>>();

        Ok(Graph { nodes, root, inputs, weights, device })
    }

    fn search_for_fusion(&self, _node: usize) -> Option<(Vec<usize>, NodeData)> {
        None
    }
}
