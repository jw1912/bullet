use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::Arc,
};

use super::{Graph, Operation, OperationQueue};
use crate::{device::Device, shape::Shape, tensor::Tensor};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Node(pub(crate) usize);

pub(crate) struct NodeData<D: Device> {
    own: Node,
    id: Option<String>,
    shape: Shape,
    requires_grad: bool,
    parent_operation: Option<Box<dyn Operation<D>>>,
    parent_nodes: Vec<Node>,
}

impl<D: Device> NodeData<D> {
    pub fn new(
        id: Option<String>,
        operation: Option<Box<dyn Operation<D>>>,
        shape: Shape,
        requires_grad: bool,
        parents: &[Node],
    ) -> Self {
        Self {
            id,
            own: Node(usize::MAX),
            shape,
            requires_grad,
            parent_operation: operation,
            parent_nodes: parents.to_vec(),
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }
}

#[derive(Default)]
pub struct GraphBuilder<D: Device> {
    nodes: Vec<NodeData<D>>,
    roots: HashSet<Node>,
    inputs: HashSet<Node>,
    weights: HashSet<Node>,
    ids: HashSet<String>,
}

impl<D: Device> GraphBuilder<D> {
    pub(crate) fn get_node(&self, index: Node) -> &NodeData<D> {
        &self.nodes[index.0]
    }

    fn create_node(&mut self, mut data: NodeData<D>) -> Node {
        assert!(data.shape.batch_size().is_none(), "Cannot specify batch size in graph builder!");

        if let Some(id) = data.id.as_ref() {
            assert!(self.ids.insert(id.to_string()))
        }

        let node = Node(self.nodes.len());
        data.own = node;

        for parent in &data.parent_nodes {
            self.roots.remove(parent);
        }

        self.nodes.push(data);
        self.roots.insert(node);

        node
    }

    pub fn create_input(&mut self, id: &str, shape: Shape) -> Node {
        let node = self.create_node(NodeData::new(Some(id.to_string()), None, shape, false, &[]));

        self.inputs.insert(node);

        node
    }

    pub fn create_weights(&mut self, id: &str, shape: Shape) -> Node {
        let node = self.create_node(NodeData::new(Some(id.to_string()), None, shape, true, &[]));

        self.weights.insert(node);

        node
    }

    pub fn create_result_of_operation(&mut self, operation: impl Operation<D>, inputs: &[Node]) -> Node {
        let mut set = HashSet::new();
        assert!(inputs.iter().all(|node| set.insert(node)), "An operation will alias nodes on backprop!");

        let input_shape = inputs.iter().map(|node| self.get_node(*node).shape).collect::<Vec<_>>();

        match operation.output_tensor(&input_shape) {
            Ok(shape) => self.create_node(NodeData::new(None, Some(Box::new(operation)), shape, true, inputs)),
            Err(s) => panic!("{s}"),
        }
    }

    pub fn root(&self) -> Node {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");
        *self.roots.iter().next().unwrap()
    }

    pub fn build(self, device: D) -> Graph<D> {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");

        let root = *self.roots.iter().next().unwrap();
        assert!(self.get_node(root).requires_grad, "Output cannot be an input!");
        assert!(!self.weights.contains(&root), "Can't output trainable weights!");
        assert_eq!(self.get_node(root).shape, Shape::new(1, 1), "Graph output must be scalar!");

        let device = Arc::new(device);

        let nodes = self
            .nodes
            .iter()
            .map(|node_data| RefCell::new(Tensor::new(device.clone(), node_data.shape, node_data.requires_grad)))
            .collect::<Vec<_>>();

        let inputs =
            self.inputs.iter().map(|&node| (self.get_node(node).id.clone().unwrap(), node)).collect::<HashMap<_, _>>();

        let weights =
            self.weights.iter().map(|&node| (self.get_node(node).id.clone().unwrap(), node)).collect::<HashMap<_, _>>();

        let mut compiled_graph = OperationQueue::default();

        for node in self.nodes {
            if let Some(operation) = node.parent_operation {
                compiled_graph.push(operation, &node.parent_nodes, node.own);
            }
        }

        Graph { nodes, root, inputs, weights, compiled_graph, device }
    }
}
