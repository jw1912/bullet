pub mod builder;
pub mod error;
pub mod operation;
pub mod tests;

use std::{cell::RefCell, collections::HashMap, sync::Arc};

use builder::Node;

use crate::{
    device::{Device, OperationError},
    tensor::Tensor,
};

pub struct Graph<D: Device> {
    nodes: Vec<RefCell<Tensor<D>>>,
    root: usize,
    inputs: HashMap<String, usize>,
    weights: HashMap<String, usize>,
    device: Arc<D>,
}

impl<D: Device> Graph<D> {
    pub fn forward(&mut self) -> Result<f32, OperationError<D::DeviceError>> {
        for node in 0..self.nodes.len() {
            let node = { self.nodes[node].borrow().own };
            self.forward_node(node)?;
        }

        Ok(self.nodes[self.root].borrow().get_scalar().unwrap())
    }

    pub fn backward(&mut self) -> Result<(), OperationError<D::DeviceError>> {
        self.nodes[self.root].get_mut().set_grad_to_unit()?;

        for node in (0..self.nodes.len()).rev() {
            let node = { self.nodes[node].borrow().own };
            self.backward_node(node)?;
        }

        Ok(())
    }

    pub fn zero_grads(&mut self) -> Result<(), D::DeviceError> {
        for node in &mut self.nodes {
            node.get_mut().zero_grad()?;
        }

        Ok(())
    }

    pub fn input_ids(&self) -> Vec<String> {
        self.inputs.keys().cloned().collect()
    }

    pub fn weight_ids(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }

    pub fn get_input(&self, id: &str) -> std::cell::Ref<'_, Tensor<D>> {
        self.nodes[self.inputs[id]].borrow()
    }

    pub fn get_input_mut(&mut self, id: &str) -> &mut Tensor<D> {
        self.nodes[self.inputs[id]].get_mut()
    }

    pub fn get_weights(&self, id: &str) -> std::cell::Ref<'_, Tensor<D>> {
        self.nodes[self.weights[id]].borrow()
    }

    pub fn get_weights_mut(&mut self, id: &str) -> &mut Tensor<D> {
        self.nodes[self.weights[id]].get_mut()
    }

    pub fn get_node(&self, node: Node) -> std::cell::Ref<'_, Tensor<D>> {
        self.nodes[node.idx].borrow()
    }

    pub fn get_num_params(&self) -> usize {
        let mut total = 0;

        for weight in self.weight_ids() {
            total += self.get_weights(&weight).values.size();
        }

        total
    }

    pub fn synchronise(&self) -> Result<(), D::DeviceError> {
        self.device.synchronise()
    }

    pub fn get_last_device_error(&self) -> Result<(), D::DeviceError> {
        self.device.get_last_device_error()
    }

    pub fn device(&self) -> Arc<D> {
        self.device.clone()
    }
}
