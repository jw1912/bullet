use std::{cell::RefCell, fmt::Debug, time::Instant};

use super::Node;
use crate::tensor::{util, ExecutionContext, Shape, Tensor};

pub trait Operation: Debug + 'static {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String>;

    fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor);

    fn backward(&self, ctx: &mut ExecutionContext, output_grad: &Tensor, inputs: &mut [&mut Tensor]);

    fn name(&self) -> String {
        format!("{:?}", self)
    }
}

pub struct OperationPayload {
    pub(super) operation: Box<dyn Operation>,
    pub(super) inputs: Vec<Node>,
    pub(super) output: Node,
    pub(super) time_spent: Option<(u128, u128)>,
}

#[derive(Default)]
pub struct OperationQueue {
    pub(super) queue: Vec<OperationPayload>,
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
