use std::cell::RefCell;

use super::Node;
use crate::{
    tensor::{Operation, Tensor},
    ExecutionContext,
};

#[derive(Debug)]
pub struct OperationPayload {
    operation: Operation,
    inputs: Vec<Node>,
    output: Node,
}

#[derive(Debug, Default)]
pub struct OperationQueue<const BACKPROP: bool> {
    queue: Vec<OperationPayload>,
}

impl<const BACKPROP: bool> OperationQueue<BACKPROP> {
    pub fn push(&mut self, operation: Operation, inputs: &[Node], output: Node) {
        self.queue.push(OperationPayload { operation, inputs: inputs.to_vec(), output });
    }
}

impl OperationQueue<false> {
    pub fn execute_on(&self, ctx: &mut ExecutionContext, graph: &mut [RefCell<Tensor>]) {
        for OperationPayload { operation, inputs, output } in &self.queue {
            let inputs = inputs.iter().map(|node| graph[node.0].borrow()).collect::<Vec<_>>();

            let inputs = inputs.iter().map(|ref_cell| &**ref_cell).collect::<Vec<_>>();

            let mut output = graph[output.0].borrow_mut();

            operation.forward(ctx, &inputs, &mut output);
        }
    }
}

impl OperationQueue<true> {
    pub fn execute_on(&self, ctx: &mut ExecutionContext, graph: &mut [RefCell<Tensor>]) {
        for OperationPayload { operation, inputs, output } in &self.queue {
            let mut inputs = inputs.iter().map(|node| graph[node.0].borrow_mut()).collect::<Vec<_>>();

            let mut inputs = inputs.iter_mut().map(|ref_cell| &mut **ref_cell).collect::<Vec<_>>();

            let output = graph[output.0].borrow();

            operation.backward(ctx, &output, &mut inputs);
        }
    }
}
