use std::{cell::RefCell, collections::HashMap, fmt::Display};

use super::{operation::OperationQueue, Node};
use crate::{tensor::Tensor, ExecutionContext};

pub struct Graph {
    nodes: Vec<RefCell<Tensor>>,
    root: Node,
    inputs: HashMap<String, Node>,
    weights: HashMap<String, Node>,
    forward: OperationQueue<false>,
    backward: OperationQueue<true>,
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
        forward: OperationQueue<false>,
        backward: OperationQueue<true>,
        execution_context: ExecutionContext,
    ) -> Self {
        Self { nodes, root, inputs, weights, forward, backward, execution_context }
    }

    pub fn forward(&mut self) -> f32 {
        self.forward.execute_on(&mut self.execution_context, &mut self.nodes);
        self.nodes[self.root.0].borrow().get_scalar().unwrap()
    }

    pub fn backward(&mut self) {
        self.nodes[self.root.0].get_mut().set_grad_to_unit();
        self.backward.execute_on(&mut self.execution_context, &mut self.nodes);
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
        self.forward.profile_all_operations();
        self.backward.profile_all_operations();
    }

    pub fn disable_profiling(&mut self) {
        self.forward.disable_profiling();
        self.backward.disable_profiling();
    }

    pub fn profile_operation_that_produces(&mut self, node: Node) {
        self.forward.profile_operation_that_produces(node);
        self.backward.profile_operation_that_produces(node);
    }

    pub fn report_profiles(&self) {
        println!("---------------------------- Profile ----------------------------");
        println!("Operation                      Fwd             Bwd");
        println!("-----------------------------------------------------------------");
        let mut count = 0;

        for (fwd, bwd) in self.forward.queue.iter().zip(self.backward.queue.iter().rev()) {
            if let Some(fwd_time) = fwd.time_spent {
                count += 1;
                let bwd_time = bwd.time_spent.unwrap();

                assert_eq!(fwd.inputs, bwd.inputs);
                assert_eq!(fwd.output, bwd.output);

                let name = format!("{:?}", fwd.operation);

                println!("{name: <30} {fwd_time: <15} {bwd_time: <15}");
            }
        }

        if count == 0 {
            println!("        No profiling data!");
        }

        println!("-----------------------------------------------------------------");
    }
}
