use std::{collections::HashMap, sync::Arc};

use crate::{
    device::{Device, OperationError},
    graph::{Graph, GraphNodeId, GraphNodeIdTy, builder::Shape},
    tensor::{DenseMatrix, Matrix, SparseMatrix},
    trainer::TrainerError,
};

pub trait DataLoader: Send + Sync + 'static {
    type Error: Send + Sync;

    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(self, batch_size: usize, f: F) -> Result<(), Self::Error>;
}

pub struct PreparedBatchHost {
    pub batch_size: usize,
    pub inputs: HashMap<String, HostMatrix>,
}

pub enum HostMatrix {
    Sparse(HostSparseMatrix),
    Dense(HostDenseMatrix),
}

pub struct HostSparseMatrix {
    vals: Vec<i32>,
    shape: Shape,
    nnz: usize,
    batch_size: Option<usize>,
}

impl HostSparseMatrix {
    /// # Safety
    /// All values must be in the range -1..shape.size()
    pub unsafe fn new(vals: Vec<i32>, batch_size: Option<usize>, shape: Shape, nnz: usize) -> Self {
        assert_eq!(batch_size.unwrap_or(1) * nnz, vals.len());

        Self { vals, shape, nnz, batch_size }
    }
}

pub struct HostDenseMatrix {
    vals: Vec<f32>,
    shape: Shape,
    batch_size: Option<usize>,
}

impl HostDenseMatrix {
    pub fn new(vals: Vec<f32>, batch_size: Option<usize>, shape: Shape) -> Self {
        assert_eq!(batch_size.unwrap_or(1) * shape.size(), vals.len());

        Self { vals, shape, batch_size }
    }
}

pub struct PreparedBatchDevice<D: Device> {
    pub device: Arc<D>,
    pub inputs: HashMap<String, Matrix<D>>,
    pub batch_size: usize,
}

impl<D: Device> PreparedBatchDevice<D> {
    pub fn new(device: Arc<D>, data: &PreparedBatchHost) -> Result<Self, D::DeviceError> {
        let mut inputs = HashMap::new();

        let mut max_batch_size = 1;

        for (id, matrix) in &data.inputs {
            let matrix = match matrix {
                HostMatrix::Sparse(HostSparseMatrix { vals, shape, nnz, batch_size }) => {
                    max_batch_size = batch_size.unwrap_or(1).max(max_batch_size);

                    let mut sparse = SparseMatrix::zeroed(device.clone(), shape.size(), *nnz, *batch_size)?;

                    // # Safety
                    // HostMatrix::Sparse is verified on construction.
                    unsafe {
                        sparse.load_from_slice(*nnz, *batch_size, vals)?;
                    }

                    Matrix::Sparse(sparse)
                }
                HostMatrix::Dense(HostDenseMatrix { vals, shape, batch_size }) => {
                    max_batch_size = batch_size.unwrap_or(1).max(max_batch_size);
                    let mut dense = DenseMatrix::zeroed(device.clone(), shape.size(), *batch_size)?;
                    dense.load_from_slice(*batch_size, vals)?;
                    Matrix::Dense(dense)
                }
            };

            inputs.insert(id.clone(), matrix);
        }

        device.synchronise()?;

        Ok(Self { device, inputs, batch_size: max_batch_size })
    }

    pub fn load_new_data(&mut self, data: &PreparedBatchHost) -> Result<(), OperationError<D::DeviceError>> {
        for (id, matrix) in &data.inputs {
            match matrix {
                HostMatrix::Sparse(HostSparseMatrix { vals, shape, nnz, batch_size }) => {
                    let sparse = self.inputs.get_mut(id).unwrap().sparse_mut()?;

                    if shape.size() != sparse.single_size || *nnz != sparse.nnz {
                        return Err(OperationError::InvalidTensorFormat);
                    }

                    // # Safety
                    // HostMatrix::Sparse is verified on construction.
                    unsafe {
                        sparse.load_non_blocking_from_host(*nnz, *batch_size, vals)?;
                    }
                }
                HostMatrix::Dense(HostDenseMatrix { vals, shape, batch_size }) => {
                    let dense = self.inputs.get_mut(id).unwrap().dense_mut()?;

                    if shape.size() != dense.single_size {
                        return Err(OperationError::InvalidTensorFormat);
                    }

                    unsafe {
                        dense.load_non_blocking_from_host(*batch_size, vals)?;
                    }
                }
            }
        }

        self.device.synchronise()?;

        Ok(())
    }

    pub fn load_into_graph(&mut self, graph: &mut Graph<D>) -> Result<(), TrainerError<D>> {
        for (id, matrix) in &mut self.inputs {
            if let Some(idx) = graph.input_idx(id) {
                let tensor = graph.get(GraphNodeId::new(idx, GraphNodeIdTy::Values)).unwrap();
                tensor.swap_with(matrix).map_err(TrainerError::Unexpected)?;
            }
        }

        Ok(())
    }
}
