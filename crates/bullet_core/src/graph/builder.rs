use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::Arc,
};

use super::{operation::Operation, Graph};
use crate::{device::Device, shape::Shape, tensor::Tensor};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Node(pub(crate) usize);

pub(crate) struct NodeData {
    id: Option<String>,
    shape: Shape,
    requires_grad: bool,
    parent_operation: Option<Operation>,
}

impl NodeData {
    pub fn new(id: Option<String>, operation: Option<Operation>, shape: Shape, requires_grad: bool) -> Self {
        Self { id, shape, requires_grad, parent_operation: operation }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }
}

#[derive(Default)]
pub struct GraphBuilder {
    nodes: Vec<NodeData>,
    roots: HashSet<Node>,
    inputs: HashSet<Node>,
    weights: HashSet<Node>,
    ids: HashSet<String>,
}

impl GraphBuilder {
    pub(crate) fn get_node(&self, index: Node) -> &NodeData {
        &self.nodes[index.0]
    }

    fn create_node(&mut self, data: NodeData) -> Node {
        assert!(data.shape.batch_size().is_none(), "Cannot specify batch size in graph builder!");

        if let Some(id) = data.id.as_ref() {
            assert!(self.ids.insert(id.to_string()))
        }

        let node = Node(self.nodes.len());

        if let Some(op) = &data.parent_operation {
            for parent in &op.nodes() {
                self.roots.remove(parent);
            }
        }

        self.nodes.push(data);
        self.roots.insert(node);

        node
    }

    pub fn create_input(&mut self, id: &str, shape: Shape) -> Node {
        let node = self.create_node(NodeData::new(Some(id.to_string()), None, shape, false));

        self.inputs.insert(node);

        node
    }

    pub fn create_weights(&mut self, id: &str, shape: Shape) -> Node {
        let node = self.create_node(NodeData::new(Some(id.to_string()), None, shape, true));

        self.weights.insert(node);

        node
    }

    pub fn create_result_of_operation(&mut self, operation: Operation) -> Node {
        match operation.output_shape(self) {
            Ok(shape) => self.create_node(NodeData::new(None, Some(operation), shape, true)),
            Err(s) => panic!("{s:?}"),
        }
    }

    pub fn root(&self) -> Node {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");
        *self.roots.iter().next().unwrap()
    }

    pub fn build<D: Device>(self, device: D) -> Graph<D> {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");

        let root = *self.roots.iter().next().unwrap();
        assert!(self.get_node(root).requires_grad, "Output cannot be an input!");
        assert!(!self.weights.contains(&root), "Can't output trainable weights!");
        assert_eq!(self.get_node(root).shape, Shape::new(1, 1), "Graph output must be scalar!");

        let device = Arc::new(device);

        let nodes = self
            .nodes
            .iter()
            .map(|node_data| {
                RefCell::new(Tensor::new(
                    device.clone(),
                    node_data.shape,
                    node_data.requires_grad,
                    node_data.parent_operation,
                ))
            })
            .collect::<Vec<_>>();

        let inputs =
            self.inputs.iter().map(|&node| (self.get_node(node).id.clone().unwrap(), node)).collect::<HashMap<_, _>>();

        let weights =
            self.weights.iter().map(|&node| (self.get_node(node).id.clone().unwrap(), node)).collect::<HashMap<_, _>>();

        Graph { nodes, root, inputs, weights, device }
    }
}
