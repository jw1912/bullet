use std::{cell::RefCell, time::Instant};

use super::Node;
use crate::{
    backend::util,
    tensor::{Operation, Tensor},
    ExecutionContext,
};

#[derive(Debug)]
pub struct OperationPayload {
    pub(super) operation: Operation,
    pub(super) inputs: Vec<Node>,
    pub(super) output: Node,
    pub(super) time_spent: Option<u128>,
}

#[derive(Debug, Default)]
pub struct OperationQueue<const BACKPROP: bool> {
    pub(super) queue: Vec<OperationPayload>,
}

impl<const BACKPROP: bool> OperationQueue<BACKPROP> {
    pub fn push(&mut self, operation: Operation, inputs: &[Node], output: Node) {
        self.queue.push(OperationPayload { operation, inputs: inputs.to_vec(), output, time_spent: None });
    }

    pub fn profile_all_operations(&mut self) {
        for op in &mut self.queue {
            op.time_spent = Some(0);
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
                op.time_spent = Some(0);
            }
        }
    }
}

impl OperationQueue<false> {
    pub fn execute_on(&mut self, ctx: &mut ExecutionContext, graph: &mut [RefCell<Tensor>]) {
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
                *spent += t.elapsed().as_micros();
            }
        }
    }
}

impl OperationQueue<true> {
    pub fn execute_on(&mut self, ctx: &mut ExecutionContext, graph: &mut [RefCell<Tensor>]) {
        for OperationPayload { operation, inputs, output, time_spent } in &mut self.queue {
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
                *spent += t.elapsed().as_micros();
            }
        }
    }
}
