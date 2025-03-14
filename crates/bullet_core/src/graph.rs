pub mod builder;
pub mod execution;
pub mod ir;
pub mod tests;

use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    fmt::Debug,
    sync::Arc,
    time::Instant,
};

use ir::{node::AnnotatedNode, GraphIRError};

use crate::backend::{
    device::{blas::Shape, Device, OperationError},
    tensor::Tensor,
};

pub struct Graph<D: Device> {
    nodes: Vec<Option<RefCell<Tensor<D>>>>,
    root: usize,
    inputs: HashMap<String, usize>,
    weights: HashMap<String, usize>,
    device: Arc<D>,
    profile: HashMap<Node, ProfileInformation>,
}

#[derive(Debug)]
pub enum GraphError<T: Debug> {
    Builder(GraphIRError),
    Operation(OperationError<T>),
    DeviceError(T),
}

impl<T: Debug> From<GraphIRError> for GraphError<T> {
    fn from(value: GraphIRError) -> Self {
        Self::Builder(value)
    }
}

impl<T: Debug> From<OperationError<T>> for GraphError<T> {
    fn from(value: OperationError<T>) -> Self {
        Self::Operation(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Node {
    idx: usize,
    pub shape: Shape,
}

impl From<AnnotatedNode> for Node {
    fn from(value: AnnotatedNode) -> Self {
        Self { idx: value.idx, shape: value.shape }
    }
}

#[derive(Clone, Default)]
struct ProfileInformation {
    name: String,
    fwd_time: u128,
    fwd_count: u128,
    bwd_time: u128,
    bwd_count: u128,
}

impl<D: Device> Graph<D> {
    fn get_node_info(&self, idx: usize) -> Result<AnnotatedNode, OperationError<D::DeviceError>> {
        if let Ok(node) = self.get(idx) {
            Ok(node.own)
        } else {
            Err(OperationError::TensorOptimisedOut)
        }
    }

    pub fn sanity_check(&self) {
        self.device().sanity_check();
    }

    pub fn get_node(&self, node: Node) -> Ref<'_, Tensor<D>> {
        self.get(node.idx).unwrap()
    }

    fn get(&self, idx: usize) -> Result<Ref<'_, Tensor<D>>, OperationError<D::DeviceError>> {
        if let Some(tensor) = &self.nodes[idx] {
            Ok(tensor.borrow())
        } else {
            Err(OperationError::UnsupportedOperation)
        }
    }

    fn get_mut(&self, idx: usize) -> Result<RefMut<'_, Tensor<D>>, OperationError<D::DeviceError>> {
        if let Some(tensor) = &self.nodes[idx] {
            Ok(tensor.borrow_mut())
        } else {
            Err(OperationError::UnsupportedOperation)
        }
    }

    pub fn forward(&mut self) -> Result<f32, OperationError<D::DeviceError>> {
        for node in 0..self.nodes.len() {
            let node = self.get_node_info(node)?;

            let t = if self.profile.contains_key(&node.into()) {
                self.device().synchronise()?;
                Some(Instant::now())
            } else {
                None
            };

            self.forward_node(node)?;

            if let Some(t) = t {
                self.device().synchronise()?;
                let prof = self.profile.get_mut(&node.into()).unwrap();
                prof.fwd_time += t.elapsed().as_micros();
                prof.fwd_count += 1;
            }
        }

        Ok(self.get(self.root)?.get_scalar().unwrap())
    }

    pub fn backward(&mut self) -> Result<(), OperationError<D::DeviceError>> {
        self.get_mut(self.root)?.set_grad_to_unit()?;

        for node in (0..self.nodes.len()).rev() {
            let node = self.get_node_info(node)?;

            let t = if self.profile.contains_key(&node.into()) {
                self.device().synchronise()?;
                Some(Instant::now())
            } else {
                None
            };

            self.backward_node(node)?;

            if let Some(t) = t {
                self.device().synchronise()?;
                let prof = self.profile.get_mut(&node.into()).unwrap();
                prof.bwd_time += t.elapsed().as_micros();
                prof.bwd_count += 1;
            }
        }

        Ok(())
    }

    pub fn profile_node(&mut self, node: Node, id: &str) {
        self.profile.insert(node, ProfileInformation { name: id.to_string(), ..Default::default() });
    }

    pub fn profile_all_nodes(&mut self) {
        for node in 0..self.nodes.len() {
            if let Some(tensor) = &self.nodes[node] {
                let tensor = tensor.borrow();
                if let Some(op) = tensor.operation.as_ref() {
                    let node = tensor.own;
                    let id = format!("{:?}", *op);
                    let id = id.split_once('(').unwrap();
                    let name = format!("Node {: >2} = {}", node.idx, id.0);
                    self.profile.insert(node.into(), ProfileInformation { name, ..Default::default() });
                }
            }
        }
    }

    pub fn report_profiles(&self) {
        let mut vals = self.profile.values().cloned().collect::<Vec<_>>();
        vals.sort_by_key(|prof| prof.fwd_time + prof.bwd_time);

        let mut fwd = 0;
        let mut bwd = 0;

        println!("+--------------- Profile ---------------- Fwd ------ Bwd -------+");

        for prof in vals.iter().rev() {
            if prof.fwd_count + prof.bwd_count > 0 {
                print!("| {: <40}", prof.name);
                if prof.fwd_count > 0 {
                    let avg = prof.fwd_time / prof.fwd_count;
                    fwd += avg;
                    print!("{avg: <10} ");
                } else {
                    print!("{: <10} ", "N/A");
                }

                if prof.bwd_count > 0 {
                    let avg = prof.bwd_time / prof.bwd_count;
                    bwd += avg;
                    println!("{avg: <10} |");
                } else {
                    println!("{: <10} |", "N/A");
                }
            }
        }

        println!("+---------------------------------------------------------------+");
        println!("| {: <40}{fwd: <10} {bwd: <10} |", "Total");
        println!("+---------------------------------------------------------------+");
    }

    pub fn zero_grads(&mut self) -> Result<(), D::DeviceError> {
        for node in &mut self.nodes {
            if let Some(node) = node.as_mut() {
                node.get_mut().zero_grad()?;
            }
        }

        Ok(())
    }

    pub fn input_ids(&self) -> Vec<String> {
        self.inputs.keys().cloned().collect()
    }

    pub fn weight_ids(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }

    pub fn get_input(&self, id: &str) -> Ref<'_, Tensor<D>> {
        self.get(self.inputs[id]).unwrap()
    }

    pub fn get_input_mut(&mut self, id: &str) -> RefMut<'_, Tensor<D>> {
        self.get_mut(self.inputs[id]).unwrap()
    }

    pub fn get_weights(&self, id: &str) -> Ref<'_, Tensor<D>> {
        self.get(self.weights[id]).unwrap()
    }

    pub fn get_weights_mut(&mut self, id: &str) -> RefMut<'_, Tensor<D>> {
        self.get_mut(self.weights[id]).unwrap()
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
