use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::{Debug, Display},
    ops::Index,
    time::Instant,
};

use crate::{
    tensor::{util, Tensor},
    ExecutionContext, Shape,
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Node(pub(crate) usize);

pub struct NodeData {
    own: Node,
    id: Option<String>,
    shape: Shape,
    requires_grad: bool,
    parent_operation: Option<Box<dyn Operation>>,
    parent_nodes: Vec<Node>,
}

impl NodeData {
    pub fn new(
        id: Option<String>,
        operation: Option<Box<dyn Operation>>,
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
}

#[derive(Default)]
pub struct GraphBuilder {
    nodes: Vec<NodeData>,
    roots: HashSet<Node>,
    inputs: HashSet<Node>,
    weights: HashSet<Node>,
    ids: HashSet<String>,
}

impl Index<Node> for GraphBuilder {
    type Output = NodeData;

    fn index(&self, index: Node) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl GraphBuilder {
    fn create_node(&mut self, mut data: NodeData) -> Node {
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

    pub fn create_result_of_operation(&mut self, operation: impl Operation, inputs: &[Node]) -> Node {
        let mut set = HashSet::new();
        assert!(inputs.iter().all(|node| set.insert(node)), "An operation will alias nodes on backprop!");

        let input_shape = inputs.iter().map(|node| self[*node].shape).collect::<Vec<_>>();

        match operation.output_tensor(&input_shape) {
            Ok(shape) => self.create_node(NodeData::new(None, Some(Box::new(operation)), shape, true, inputs)),
            Err(s) => panic!("{s}"),
        }
    }

    pub fn build(self, execution_context: ExecutionContext) -> Graph {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");

        let root = *self.roots.iter().next().unwrap();
        assert!(self[root].requires_grad, "Output cannot be an input!");
        assert!(!self.weights.contains(&root), "Can't output trainable weights!");

        let nodes = self
            .nodes
            .iter()
            .map(|node_data| RefCell::new(Tensor::new(node_data.shape, node_data.requires_grad)))
            .collect::<Vec<_>>();

        let inputs = self.inputs.iter().map(|&node| (self[node].id.clone().unwrap(), node)).collect::<HashMap<_, _>>();

        let weights =
            self.weights.iter().map(|&node| (self[node].id.clone().unwrap(), node)).collect::<HashMap<_, _>>();

        let mut compiled_graph = OperationQueue::default();

        for node in self.nodes {
            if let Some(operation) = node.parent_operation {
                compiled_graph.push(operation, &node.parent_nodes, node.own);
            }
        }

        Graph::new(nodes, root, inputs, weights, compiled_graph, execution_context)
    }
}

pub struct Graph {
    nodes: Vec<RefCell<Tensor>>,
    root: Node,
    inputs: HashMap<String, Node>,
    weights: HashMap<String, Node>,
    compiled_graph: OperationQueue,
    execution_context: ExecutionContext,
}

impl Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.nodes)
    }
}

impl Graph {
    pub fn new(
        nodes: Vec<RefCell<Tensor>>,
        root: Node,
        inputs: HashMap<String, Node>,
        weights: HashMap<String, Node>,
        compiled_graph: OperationQueue,
        execution_context: ExecutionContext,
    ) -> Self {
        Self { nodes, root, inputs, weights, compiled_graph, execution_context }
    }

    pub fn forward(&mut self) -> f32 {
        self.compiled_graph.execute_fwd(&mut self.execution_context, &mut self.nodes);
        self.nodes[self.root.0].borrow().get_scalar().unwrap()
    }

    pub fn backward(&mut self) {
        self.nodes[self.root.0].get_mut().set_grad_to_unit();
        self.compiled_graph.execute_bwd(&mut self.execution_context, &mut self.nodes);
    }

    fn store_values(&mut self, node: Node, data: &Tensor) {
        data.copy_values_into(self.nodes[node.0].get_mut());
    }

    pub fn store_input(&mut self, input: &str, data: &Tensor) {
        self.store_values(self.inputs[input], data);
    }

    pub fn store_weights(&mut self, weights: &str, data: &Tensor) {
        self.store_values(self.weights[weights], data);
    }

    pub fn zero_grads(&mut self) {
        for node in &mut self.nodes {
            node.get_mut().zero_grad();
        }
    }

    pub fn input_ids(&self) -> Vec<String> {
        self.inputs.keys().cloned().collect()
    }

    pub fn weight_ids(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }

    pub fn get_input(&self, id: &str) -> std::cell::Ref<'_, Tensor> {
        self.nodes[self.inputs[id].0].borrow()
    }

    pub fn get_input_mut(&mut self, id: &str) -> &mut Tensor {
        self.nodes[self.inputs[id].0].get_mut()
    }

    pub fn get_weights(&self, id: &str) -> std::cell::Ref<'_, Tensor> {
        self.nodes[self.weights[id].0].borrow()
    }

    pub fn get_weights_mut(&mut self, id: &str) -> &mut Tensor {
        self.nodes[self.weights[id].0].get_mut()
    }

    pub fn get_node(&self, node: Node) -> std::cell::Ref<'_, Tensor> {
        self.nodes[node.0].borrow()
    }

    pub fn get_num_params(&self) -> usize {
        let mut total = 0;

        for weight in self.weight_ids() {
            total += self.get_weights(&weight).values.shape().size();
        }

        total
    }

    pub fn profile_all_operations(&mut self) {
        self.compiled_graph.profile_all_operations();
    }

    pub fn disable_profiling(&mut self) {
        self.compiled_graph.disable_profiling();
    }

    pub fn profile_operation_that_produces(&mut self, node: Node) {
        self.compiled_graph.profile_operation_that_produces(node);
    }

    pub fn report_profiles(&self) {
        println!("---------------------------- Profile ----------------------------");
        println!("Operation                      Fwd             Bwd");
        println!("-----------------------------------------------------------------");
        let mut count = 0;

        for operation in &self.compiled_graph.queue {
            if let Some((fwd_time, bwd_time)) = operation.time_spent {
                count += 1;
                println!("{: <30} {fwd_time: <15} {bwd_time: <15}", operation.operation.name());
            }
        }

        if count == 0 {
            println!("No profiling data!");
        }

        println!("-----------------------------------------------------------------");
    }
}

pub trait Operation: Debug + 'static {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String>;

    fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor);

    fn backward(&self, ctx: &mut ExecutionContext, output_grad: &Tensor, inputs: &mut [&mut Tensor]);

    fn name(&self) -> String {
        format!("{:?}", self)
    }
}

pub struct OperationPayload {
    operation: Box<dyn Operation>,
    inputs: Vec<Node>,
    output: Node,
    time_spent: Option<(u128, u128)>,
}

#[derive(Default)]
pub struct OperationQueue {
    queue: Vec<OperationPayload>,
}

impl OperationQueue {
    pub fn push(&mut self, operation: Box<dyn Operation>, inputs: &[Node], output: Node) {
        self.queue.push(OperationPayload { operation, inputs: inputs.to_vec(), output, time_spent: None });
    }

    pub fn profile_all_operations(&mut self) {
        for op in &mut self.queue {
            op.time_spent = Some((0, 0));
        }
    }

    pub fn disable_profiling(&mut self) {
        for op in &mut self.queue {
            op.time_spent = None;
        }
    }

    pub fn profile_operation_that_produces(&mut self, node: Node) {
        for op in &mut self.queue {
            if op.output == node {
                op.time_spent = Some((0, 0));
            }
        }
    }

    pub fn execute_fwd(&mut self, ctx: &mut ExecutionContext, graph: &mut [RefCell<Tensor>]) {
        for OperationPayload { operation, inputs, output, time_spent } in &mut self.queue {
            if time_spent.is_some() {
                util::device_synchronise();
            }
            let t = Instant::now();

            let inputs = inputs.iter().map(|node| graph[node.0].borrow()).collect::<Vec<_>>();

            let inputs = inputs.iter().map(|ref_cell| &**ref_cell).collect::<Vec<_>>();

            let mut output = graph[output.0].borrow_mut();

            operation.forward(ctx, &inputs, &mut output);

            if let Some(spent) = time_spent {
                util::device_synchronise();
                spent.0 += t.elapsed().as_micros();
            }
        }
    }

    pub fn execute_bwd(&mut self, ctx: &mut ExecutionContext, graph: &mut [RefCell<Tensor>]) {
        for OperationPayload { operation, inputs, output, time_spent } in self.queue.iter_mut().rev() {
            if time_spent.is_some() {
                util::device_synchronise();
            }
            let t = Instant::now();

            let mut inputs = inputs.iter().map(|node| graph[node.0].borrow_mut()).collect::<Vec<_>>();

            let mut inputs = inputs.iter_mut().map(|ref_cell| &mut **ref_cell).collect::<Vec<_>>();

            let output = graph[output.0].borrow();

            operation.backward(ctx, &output, &mut inputs);

            if let Some(spent) = time_spent {
                util::device_synchronise();
                spent.1 += t.elapsed().as_micros();
            }
        }
    }
}
