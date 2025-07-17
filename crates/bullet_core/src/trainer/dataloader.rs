use std::{collections::HashMap, sync::Arc};

use crate::{
    device::{Device, OperationError},
    graph::{
        builder::Shape,
        tensor::{DenseMatrix, Matrix, SparseMatrix},
        Graph, NodeId, NodeIdTy,
    },
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
}

impl HostSparseMatrix {
    /// # Safety
    /// All values must be in the range -1..shape.size()
    pub unsafe fn new(vals: Vec<i32>, batch_size: usize, shape: Shape, nnz: usize) -> Self {
        assert_eq!(batch_size * nnz, vals.len());

        Self { vals, shape, nnz }
    }
}

pub struct HostDenseMatrix {
    vals: Vec<f32>,
    shape: Shape,
}

impl HostDenseMatrix {
    pub fn new(vals: Vec<f32>, batch_size: usize, shape: Shape) -> Self {
        assert_eq!(batch_size * shape.size(), vals.len());

        Self { vals, shape }
    }
}

pub struct PreparedBatchDevice<D: Device> {
    pub device: Arc<D>,
    pub batch_size: usize,
    pub inputs: HashMap<String, Matrix<D>>,
}

impl<D: Device> PreparedBatchDevice<D> {
    pub fn new(device: Arc<D>, data: &PreparedBatchHost) -> Result<Self, D::DeviceError> {
        let batch_size = data.batch_size;

        let mut inputs = HashMap::new();

        for (id, matrix) in &data.inputs {
            let matrix = match matrix {
                HostMatrix::Sparse(HostSparseMatrix { vals, shape, nnz }) => {
                    let mut sparse = SparseMatrix::zeroed(device.clone(), shape.size(), *nnz)?;

                    // # Safety
                    // HostMatrix::Sparse is verified on construction.
                    unsafe {
                        sparse.load_from_slice(*nnz, Some(batch_size), vals)?;
                    }

                    Matrix::Sparse(sparse)
                }
                HostMatrix::Dense(HostDenseMatrix { vals, shape }) => {
                    let mut dense = DenseMatrix::zeroed(device.clone(), shape.size())?;
                    dense.load_from_slice(Some(batch_size), vals)?;
                    Matrix::Dense(dense)
                }
            };

            inputs.insert(id.clone(), matrix);
        }

        device.synchronise()?;

        Ok(Self { device, batch_size, inputs })
    }

    pub fn load_new_data(&mut self, data: &PreparedBatchHost) -> Result<(), OperationError<D::DeviceError>> {
        let batch_size = data.batch_size;

        for (id, matrix) in &data.inputs {
            match matrix {
                HostMatrix::Sparse(HostSparseMatrix { vals, shape, nnz }) => {
                    let sparse = self.inputs.get_mut(id).unwrap().sparse_mut()?;

                    if shape.size() != sparse.single_size || *nnz != sparse.nnz {
                        return Err(OperationError::InvalidTensorFormat);
                    }

                    // # Safety
                    // HostMatrix::Sparse is verified on construction.
                    unsafe {
                        sparse.load_non_blocking_from_host(*nnz, Some(batch_size), vals)?;
                    }
                }
                HostMatrix::Dense(HostDenseMatrix { vals, shape }) => {
                    let dense = self.inputs.get_mut(id).unwrap().dense_mut()?;

                    if shape.size() != dense.single_size {
                        return Err(OperationError::InvalidTensorFormat);
                    }

                    unsafe {
                        dense.load_non_blocking_from_host(Some(batch_size), vals)?;
                    }
                }
            }
        }

        self.batch_size = batch_size;

        self.device.synchronise()?;

        Ok(())
    }

    pub fn load_into_graph(&mut self, graph: &mut Graph<D>) -> Result<(), TrainerError<D>> {
        let batch_size = self.batch_size;

        for (id, matrix) in &mut self.inputs {
            assert_eq!(batch_size, matrix.batch_size().unwrap_or(1));

            if let Some(idx) = graph.input_idx(id) {
                matrix
                    .swap_with(&mut graph.get_mut(NodeId::new(idx, NodeIdTy::Values)).unwrap().values)
                    .map_err(TrainerError::Unexpected)?;
            }
        }

        Ok(())
    }
}
