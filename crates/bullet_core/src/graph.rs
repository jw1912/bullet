pub mod builder;
pub mod operation;

use std::{cell::RefCell, collections::HashMap, fmt::Debug, sync::Arc, time::Instant};

use builder::Node;

use crate::{device::Device, shape::Shape, tensor::Tensor};

pub struct Graph<D: Device> {
    nodes: Vec<RefCell<Tensor<D>>>,
    root: Node,
    inputs: HashMap<String, Node>,
    weights: HashMap<String, Node>,
    compiled_graph: OperationQueue<D>,
    device: Arc<D>,
}

impl<D: Device> Graph<D> {
    pub fn forward(&mut self) -> f32 {
        self.compiled_graph.execute_fwd(&self.device, &mut self.nodes);
        self.nodes[self.root.0].borrow().get_scalar().unwrap()
    }

    pub fn backward(&mut self) {
        self.nodes[self.root.0].get_mut().set_grad_to_unit();
        self.compiled_graph.execute_bwd(&self.device, &mut self.nodes);
    }

    fn store_values(&mut self, node: Node, data: &Tensor<D>) {
        data.copy_values_into(self.nodes[node.0].get_mut());
    }

    pub fn store_input(&mut self, input: &str, data: &Tensor<D>) {
        self.store_values(self.inputs[input], data);
    }

    pub fn store_weights(&mut self, weights: &str, data: &Tensor<D>) {
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

    pub fn get_input(&self, id: &str) -> std::cell::Ref<'_, Tensor<D>> {
        self.nodes[self.inputs[id].0].borrow()
    }

    pub fn get_input_mut(&mut self, id: &str) -> &mut Tensor<D> {
        self.nodes[self.inputs[id].0].get_mut()
    }

    pub fn get_weights(&self, id: &str) -> std::cell::Ref<'_, Tensor<D>> {
        self.nodes[self.weights[id].0].borrow()
    }

    pub fn get_weights_mut(&mut self, id: &str) -> &mut Tensor<D> {
        self.nodes[self.weights[id].0].get_mut()
    }

    pub fn get_node(&self, node: Node) -> std::cell::Ref<'_, Tensor<D>> {
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
            if let Some((fwd_time, bwd_time, fwd_exes, bwd_exes)) = operation.time_spent {
                count += 1;
                let fwd_avg = fwd_time / u128::from(fwd_exes);
                let bwd_avg = bwd_time / u128::from(bwd_exes);
                let name = operation.operation.name();
                println!("{name: <30} {fwd_avg: <15} {bwd_avg: <15}");
            }
        }

        if count == 0 {
            println!("No profiling data!");
        }

        println!("-----------------------------------------------------------------");
    }

    pub fn synchronise(&self) {
        self.device.synchronise();
    }

    pub fn panic_if_device_error(&self, msg: &str) {
        self.device.panic_if_device_error(msg);
    }

    pub fn device(&self) -> Arc<D> {
        self.device.clone()
    }
}

pub trait Operation<D: Device>: Debug + 'static {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String>;

    fn forward(&self, inputs: &[&Tensor<D>], output: &mut Tensor<D>);

    fn backward(&self, output_grad: &Tensor<D>, inputs: &mut [&mut Tensor<D>]);

    fn name(&self) -> String {
        format!("{:?}", self)
    }
}

pub struct OperationPayload<D: Device> {
    operation: Box<dyn Operation<D>>,
    inputs: Vec<Node>,
    output: Node,
    time_spent: Option<(u128, u128, u64, u64)>,
}

pub struct OperationQueue<D: Device> {
    queue: Vec<OperationPayload<D>>,
}

impl<D: Device> Default for OperationQueue<D> {
    fn default() -> Self {
        Self { queue: Vec::new() }
    }
}

impl<D: Device> OperationQueue<D> {
    pub fn push(&mut self, operation: Box<dyn Operation<D>>, inputs: &[Node], output: Node) {
        self.queue.push(OperationPayload { operation, inputs: inputs.to_vec(), output, time_spent: None });
    }

    pub fn profile_all_operations(&mut self) {
        for op in &mut self.queue {
            op.time_spent = Some((0, 0, 0, 0));
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
                op.time_spent = Some((0, 0, 0, 0));
            }
        }
    }

    pub fn execute_fwd(&mut self, device: &D, graph: &mut [RefCell<Tensor<D>>]) {
        for OperationPayload { operation, inputs, output, time_spent } in &mut self.queue {
            if time_spent.is_some() {
                device.synchronise();
            }
            let t = Instant::now();

            let inputs = inputs.iter().map(|node| graph[node.0].borrow()).collect::<Vec<_>>();

            let inputs = inputs.iter().map(|ref_cell| &**ref_cell).collect::<Vec<_>>();

            let mut output = graph[output.0].borrow_mut();

            operation.forward(&inputs, &mut output);

            if let Some(spent) = time_spent {
                device.synchronise();
                spent.0 += t.elapsed().as_micros();
                spent.2 += 1;
            }
        }
    }

    pub fn execute_bwd(&mut self, device: &D, graph: &mut [RefCell<Tensor<D>>]) {
        for OperationPayload { operation, inputs, output, time_spent } in self.queue.iter_mut().rev() {
            if time_spent.is_some() {
                device.synchronise();
            }
            let t = Instant::now();

            let mut inputs = inputs.iter().map(|node| graph[node.0].borrow_mut()).collect::<Vec<_>>();

            let mut inputs = inputs.iter_mut().map(|ref_cell| &mut **ref_cell).collect::<Vec<_>>();

            let output = graph[output.0].borrow();

            operation.backward(&output, &mut inputs);

            if let Some(spent) = time_spent {
                device.synchronise();
                spent.1 += t.elapsed().as_micros();
                spent.3 += 1;
            }
        }
    }
}
