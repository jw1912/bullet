pub mod builder;
pub mod instruction;
pub mod ir;
pub mod tensor;

use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, Mutex},
    time::Instant,
};

use instruction::GraphInstruction;
use ir::{node::AnnotatedNode, shape::Shape, GraphIRError};
use tensor::{read_from_byte_buffer, Tensor};

use crate::device::{Device, OperationError};

pub struct GraphFunction<D: Device> {
    instructions: Vec<Box<dyn GraphInstruction<D>>>,
}

impl<D: Device> Default for GraphFunction<D> {
    fn default() -> Self {
        Self { instructions: Vec::new() }
    }
}

impl<D: Device> GraphFunction<D> {
    pub fn push(&mut self, instruction: impl GraphInstruction<D>) {
        self.instructions.push(Box::new(instruction));
    }

    pub fn extend(&mut self, rhs: Self) {
        for instr in rhs.instructions {
            self.instructions.push(instr);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeIdTy {
    Values,
    Gradients,
    Ancillary(u16),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId {
    id: usize,
    ty: NodeIdTy,
}

impl NodeId {
    pub fn new(id: usize, ty: NodeIdTy) -> Self {
        Self { id, ty }
    }
}

impl std::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Id({}, {:?})", self.id, self.ty)
    }
}

#[derive(Clone, Copy)]
struct ProfileInfo {
    executions: usize,
    total_time_micros: u128,
}

pub struct Graph<D: Device> {
    nodes: HashMap<NodeId, RefCell<Tensor<D>>>,
    inputs: HashMap<String, usize>,
    weights: HashMap<String, usize>,
    functions: HashMap<String, GraphFunction<D>>,
    profiles: HashMap<String, Mutex<Vec<ProfileInfo>>>,
    device: Arc<D>,
    root: usize,
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

impl Node {
    pub fn idx(&self) -> usize {
        self.idx
    }
}

impl From<AnnotatedNode> for Node {
    fn from(value: AnnotatedNode) -> Self {
        Self { idx: value.idx, shape: value.shape }
    }
}

impl<D: Device> Graph<D> {
    pub fn sanity_check(&self) {
        self.device().sanity_check();
    }

    pub fn get_node_values(&self, node: Node) -> Ref<'_, Tensor<D>> {
        self.get(NodeId::new(node.idx, NodeIdTy::Values)).unwrap()
    }

    pub fn get(&self, id: NodeId) -> Result<Ref<'_, Tensor<D>>, OperationError<D::DeviceError>> {
        if let Some(tensor) = &self.nodes.get(&id) {
            Ok(tensor.borrow())
        } else {
            println!("Cant find: {id:?}");
            Err(OperationError::TensorOptimisedOut)
        }
    }

    pub fn get_mut(&self, id: NodeId) -> Result<RefMut<'_, Tensor<D>>, OperationError<D::DeviceError>> {
        if let Some(tensor) = &self.nodes.get(&id) {
            Ok(tensor.borrow_mut())
        } else {
            Err(OperationError::TensorOptimisedOut)
        }
    }

    pub fn get_weights(&self, id: &str) -> Ref<'_, Tensor<D>> {
        let idx = self.weight_idx(id).unwrap();
        self.get(NodeId::new(idx, NodeIdTy::Values)).unwrap()
    }

    pub fn get_weights_mut(&self, id: &str) -> RefMut<'_, Tensor<D>> {
        let idx = self.weight_idx(id).unwrap();
        self.get_mut(NodeId::new(idx, NodeIdTy::Values)).unwrap()
    }

    pub fn get_input(&self, id: &str) -> Ref<'_, Tensor<D>> {
        let idx = self.input_idx(id).unwrap();
        self.get(NodeId::new(idx, NodeIdTy::Values)).unwrap()
    }

    pub fn get_input_mut(&self, id: &str) -> RefMut<'_, Tensor<D>> {
        let idx = self.input_idx(id).unwrap();
        self.get_mut(NodeId::new(idx, NodeIdTy::Values)).unwrap()
    }

    fn root(&self) -> usize {
        self.root
    }

    pub fn profile_function(&mut self, id: &str) -> Result<(), OperationError<D::DeviceError>> {
        let func = self.functions.get(id).ok_or(OperationError::UnsupportedOperation)?;

        let _ = self.profiles.insert(
            id.to_string(),
            Mutex::new(vec![ProfileInfo { executions: 0, total_time_micros: 0 }; func.instructions.len()]),
        );

        Ok(())
    }

    pub fn display_profile(&self, id: &str) -> Result<(), OperationError<D::DeviceError>> {
        let func = self.functions.get(id).ok_or(OperationError::UnsupportedOperation)?;
        let profile = self.profiles.get(id).ok_or(OperationError::UnsupportedOperation)?;

        println!("Profile for function '{id}':");
        for (instr, info) in func.instructions.iter().zip(profile.lock().unwrap().iter()) {
            let avg_time = info.total_time_micros / info.executions as u128;
            let dbg = format!("{instr:?}");
            let instr_name = dbg.split_whitespace().next().unwrap();

            println!("{avg_time: >6} micros for {instr_name}");
        }

        Ok(())
    }

    pub fn display_function_code(&self, id: &str) -> Result<(), OperationError<D::DeviceError>> {
        for instr in &self.functions.get(id).ok_or(OperationError::UnsupportedOperation)?.instructions {
            println!("{instr:?}");
        }

        Ok(())
    }

    pub fn execute(&mut self, id: &str) -> Result<(), OperationError<D::DeviceError>> {
        let func = self.functions.get(id).ok_or(OperationError::UnsupportedOperation)?;

        if let Some(profile) = self.profiles.get(id) {
            for (instr, info) in func.instructions.iter().zip(profile.lock().unwrap().iter_mut()) {
                if let Err(e) = self.device().synchronise() {
                    println!("Error {e:?} in function '{id}' before executing {instr:?}");
                    return Err(OperationError::DeviceError(Box::new(e)));
                }
                let t = Instant::now();

                if let Err(e) = instr.execute(self) {
                    println!("Error {e:?} in function '{id}' executing {instr:?}");
                    return Err(e);
                }

                if let Err(e) = self.device().synchronise() {
                    println!("Error {e:?} in function '{id}' after executing {instr:?}");
                    return Err(OperationError::DeviceError(Box::new(e)));
                }

                info.executions += 1;
                info.total_time_micros += t.elapsed().as_micros();
            }
        } else {
            for instr in &func.instructions {
                if let Err(e) = instr.execute(self) {
                    println!("Error {e:?} in function '{id}' executing {instr:?}");
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    pub fn forward(&mut self) -> Result<f32, OperationError<D::DeviceError>> {
        self.execute("forward")?;
        self.device.synchronise()?;
        self.device.get_last_device_error()?;
        self.get_output_val()
    }

    pub fn backward(&mut self) -> Result<(), OperationError<D::DeviceError>> {
        self.execute("backward")?;
        self.device.synchronise()?;
        self.device.get_last_device_error()?;
        Ok(())
    }

    pub fn zero_grads(&mut self) -> Result<(), OperationError<D::DeviceError>> {
        self.execute("zero_grads")?;
        self.device.synchronise()?;
        self.device.get_last_device_error()?;
        Ok(())
    }

    pub fn get_output_val(&self) -> Result<f32, OperationError<D::DeviceError>> {
        Ok(self.get(NodeId::new(self.root(), NodeIdTy::Values))?.get_scalar().unwrap())
    }

    /// Writes the weights of a graph to a file. If `gradients` is true,
    /// it will instead write the gradients of those weights.
    pub fn write_to_file(&self, path: &str) {
        use std::{fs::File, io::Write};

        let weight_ids = self.weight_ids();

        let mut buf = Vec::new();

        for id in &weight_ids {
            let idx = *self.weights.get(id).unwrap();
            let weights = self.get_mut(NodeId::new(idx, NodeIdTy::Values)).unwrap();
            let this_buf = weights.dense().unwrap().write_to_byte_buffer(id).unwrap();

            buf.extend_from_slice(&this_buf);
        }

        let mut file = File::create(path).unwrap();
        file.write_all(&buf).unwrap();
    }

    /// Loads the weights of a graph from a file. If `gradients` is true,
    /// it will instead load the gradients of those weights.
    pub fn load_from_file(&mut self, path: &str, old_format: bool) -> Result<(), OperationError<D::DeviceError>> {
        use std::{fs::File, io::Read};

        let mut buf = Vec::new();
        let mut file = File::open(path).unwrap();
        file.read_to_end(&mut buf).unwrap();

        let weight_ids = self.weight_ids();

        let mut offset = 0;

        while offset < buf.len() {
            let (buffer, id, bytes_read) = read_from_byte_buffer(&buf[offset..], old_format);

            if !weight_ids.contains(&id) {
                return Err(OperationError::NoWeightWithID(id));
            }

            let idx = *self.weights.get(&id).unwrap();
            let mut weights = self.get_mut(NodeId::new(idx, NodeIdTy::Values)).unwrap();
            let weights = weights.dense_mut().unwrap();
            let exp_size = weights.size();

            if buffer.len() != exp_size {
                return Err(OperationError::WeightLoadingError(id, Some((buffer.len(), exp_size))));
            }

            if weights.load_from_slice(None, &buffer).is_err() {
                return Err(OperationError::WeightLoadingError(id, None));
            }

            offset += bytes_read;
        }

        Ok(())
    }

    pub fn input_ids(&self) -> Vec<String> {
        self.inputs.keys().cloned().collect()
    }

    pub fn weight_ids(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }

    pub fn input_idx(&self, id: &str) -> Option<usize> {
        self.inputs.get(id).copied()
    }

    pub fn weight_idx(&self, id: &str) -> Option<usize> {
        self.weights.get(id).copied()
    }

    pub fn get_num_params(&self) -> usize {
        let mut total = 0;

        for weight in self.weight_ids() {
            let idx = *self.weights.get(&weight).unwrap();
            total += self.get(NodeId::new(idx, NodeIdTy::Values)).unwrap().dense().unwrap().size();
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
