pub mod builder;
pub mod ir;
pub mod like;
pub mod multi;
pub mod save;

use std::{collections::HashMap, fmt::Debug, sync::Arc};

use ir::{GraphIRError, node::AnnotatedNode};

use crate::{
    dag::NodeId,
    device::{
        Device, OperationError,
        function::DeviceFunction,
        tensor::{Shape, TensorRef, read_from_byte_buffer},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphNodeIdTy {
    Values,
    Gradients,
    Ancillary(u16),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphNodeId {
    id: NodeId,
    ty: GraphNodeIdTy,
}

impl GraphNodeId {
    pub fn new(id: NodeId, ty: GraphNodeIdTy) -> Self {
        Self { id, ty }
    }
}

impl std::fmt::Debug for GraphNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Id({}, {:?})", self.id.inner(), self.ty)
    }
}

pub struct Graph<D: Device> {
    nodes: HashMap<GraphNodeId, TensorRef<D>>,
    inputs: HashMap<String, NodeId>,
    weights: HashMap<String, NodeId>,
    functions: HashMap<String, DeviceFunction<D>>,
    device: Arc<D>,
    root: NodeId,
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
    idx: NodeId,
    pub shape: Shape,
}

impl Node {
    pub fn idx(&self) -> NodeId {
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

    pub fn get_node_values(&self, node: Node) -> TensorRef<D> {
        self.get(GraphNodeId::new(node.idx, GraphNodeIdTy::Values)).unwrap()
    }

    pub fn get_ref(&self, id: NodeId, ty: GraphNodeIdTy) -> TensorRef<D> {
        self.nodes.get(&GraphNodeId::new(id, ty)).cloned().unwrap()
    }

    pub fn maybe_get_ref(&self, id: NodeId, ty: GraphNodeIdTy) -> Option<TensorRef<D>> {
        self.nodes.get(&GraphNodeId::new(id, ty)).cloned()
    }

    pub fn get(&self, id: GraphNodeId) -> Result<TensorRef<D>, OperationError<D::DeviceError>> {
        if let Some(tensor) = self.nodes.get(&id) {
            Ok(tensor.clone())
        } else {
            println!("Cant find: {id:?}");
            Err(OperationError::TensorOptimisedOut)
        }
    }

    pub fn get_weights(&self, id: &str) -> TensorRef<D> {
        let idx = self.weight_idx(id).unwrap();
        self.get(GraphNodeId::new(idx, GraphNodeIdTy::Values)).unwrap()
    }

    pub fn get_input(&self, id: &str) -> TensorRef<D> {
        let idx = self.input_idx(id).unwrap();
        self.get(GraphNodeId::new(idx, GraphNodeIdTy::Values)).unwrap()
    }

    fn root(&self) -> NodeId {
        self.root
    }

    pub fn profile_function(&mut self, _id: &str) -> Result<(), OperationError<D::DeviceError>> {
        todo!();
    }

    pub fn display_profile(&self, _id: &str) -> Result<(), OperationError<D::DeviceError>> {
        todo!();
    }

    pub fn display_function_code(&self, id: &str) -> Result<(), OperationError<D::DeviceError>> {
        let func = self.functions.get(id).ok_or(OperationError::UnsupportedOperation)?;

        println!("{func}");

        Ok(())
    }

    pub fn execute(&mut self, id: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.functions.get(id).ok_or(OperationError::UnsupportedOperation)?.execute()
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
        Ok(self.get(GraphNodeId::new(self.root(), GraphNodeIdTy::Values))?.get_scalar().unwrap())
    }

    /// Writes the weights of a graph to a file. If `gradients` is true,
    /// it will instead write the gradients of those weights.
    pub fn write_to_file(&self, path: &str) {
        use std::{fs::File, io::Write};

        let weight_ids = self.weight_ids();

        let mut buf = Vec::new();

        for id in &weight_ids {
            let idx = *self.weights.get(id).unwrap();
            let weights = self.get(GraphNodeId::new(idx, GraphNodeIdTy::Values)).unwrap();
            let this_buf = weights.dense().write_to_byte_buffer(id).unwrap();

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
            let weights = self.get(GraphNodeId::new(idx, GraphNodeIdTy::Values)).unwrap();
            let mut weights = weights.dense_mut();
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

    pub fn input_idx(&self, id: &str) -> Option<NodeId> {
        self.inputs.get(id).copied()
    }

    pub fn weight_idx(&self, id: &str) -> Option<NodeId> {
        self.weights.get(id).copied()
    }

    pub fn get_num_params(&self) -> usize {
        let mut total = 0;

        for weight in self.weight_ids() {
            let idx = *self.weights.get(&weight).unwrap();
            total += self.get(GraphNodeId::new(idx, GraphNodeIdTy::Values)).unwrap().dense().size();
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
