use std::{collections::HashMap, sync::Arc};

use crate::{
    device::{
        Device, OperationError,
        tensor::{DenseMatrix, Matrix, Shape, SparseMatrix},
    },
    graph::{GraphNodeId, GraphNodeIdTy, like::GraphLike},
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
    pub devices: Vec<Arc<D>>,
    pub inputs: HashMap<String, Vec<Matrix<D>>>,
    pub batch_size: usize,
}

fn make_sparse<D: Device>(
    device: Arc<D>,
    shape: Shape,
    nnz: usize,
    batch_size: Option<usize>,
    vals: &[i32],
) -> Result<Matrix<D>, D::DeviceError> {
    let mut sparse = SparseMatrix::zeroed(device.clone(), shape.size(), nnz, batch_size)?;

    unsafe {
        sparse.load_from_slice(nnz, batch_size, vals)?;
    }

    Ok(Matrix::Sparse(sparse))
}

unsafe fn load_sparse_nonblocking<D: Device>(
    sparse: &mut SparseMatrix<D>,
    shape: Shape,
    nnz: usize,
    batch_size: Option<usize>,
    vals: &[i32],
) -> Result<(), OperationError<D::DeviceError>> {
    if shape.size() != sparse.single_size() || nnz != sparse.nnz() {
        return Err(OperationError::InvalidTensorFormat);
    }

    unsafe {
        sparse.load_non_blocking_from_host(nnz, batch_size, vals)?;
    }

    Ok(())
}

fn make_dense<D: Device>(
    device: Arc<D>,
    shape: Shape,
    batch_size: Option<usize>,
    vals: &[f32],
) -> Result<Matrix<D>, D::DeviceError> {
    let mut dense = DenseMatrix::zeroed(device.clone(), shape.size(), batch_size)?;
    dense.load_from_slice(batch_size, vals)?;
    Ok(Matrix::Dense(dense))
}

unsafe fn load_dense_nonblocking<D: Device>(
    dense: &mut DenseMatrix<D>,
    shape: Shape,
    batch_size: Option<usize>,
    vals: &[f32],
) -> Result<(), OperationError<D::DeviceError>> {
    if shape.size() != dense.single_size() {
        return Err(OperationError::InvalidTensorFormat);
    }

    unsafe {
        dense.load_non_blocking_from_host(batch_size, vals)?;
    }

    Ok(())
}

impl<D: Device> PreparedBatchDevice<D> {
    pub fn new(devices: Vec<Arc<D>>, data: &PreparedBatchHost) -> Result<Self, D::DeviceError> {
        let mut inputs = HashMap::new();

        let mut max_batch_size = 1;

        for (id, matrix) in &data.inputs {
            let matrix = match matrix {
                HostMatrix::Sparse(HostSparseMatrix { vals, shape, nnz, batch_size }) => {
                    max_batch_size = batch_size.unwrap_or(1).max(max_batch_size);

                    if let Some(b) = *batch_size {
                        let chunk_size = b.div_ceil(devices.len());
                        devices
                            .iter()
                            .zip(vals.chunks(nnz * chunk_size))
                            .map(|(d, chunk)| make_sparse(d.clone(), *shape, *nnz, Some(chunk.len() / nnz), chunk))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        devices
                            .iter()
                            .map(|d| make_sparse(d.clone(), *shape, *nnz, None, vals))
                            .collect::<Result<Vec<_>, _>>()?
                    }
                }
                HostMatrix::Dense(HostDenseMatrix { vals, shape, batch_size }) => {
                    max_batch_size = batch_size.unwrap_or(1).max(max_batch_size);

                    if let Some(b) = *batch_size {
                        let chunk_size = b.div_ceil(devices.len());
                        devices
                            .iter()
                            .zip(vals.chunks(shape.size() * chunk_size))
                            .map(|(d, chunk)| make_dense(d.clone(), *shape, Some(chunk.len() / shape.size()), chunk))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        devices
                            .iter()
                            .map(|d| make_dense(d.clone(), *shape, None, vals))
                            .collect::<Result<Vec<_>, _>>()?
                    }
                }
            };

            inputs.insert(id.clone(), matrix);
        }

        for d in &devices {
            d.synchronise()?;
        }

        Ok(Self { devices, inputs, batch_size: max_batch_size })
    }

    pub fn load_new_data(&mut self, data: &PreparedBatchHost) -> Result<(), OperationError<D::DeviceError>> {
        for (id, matrix) in &data.inputs {
            match matrix {
                HostMatrix::Sparse(HostSparseMatrix { vals, shape, nnz, batch_size }) => {
                    let sparses = self.inputs.get_mut(id).unwrap();

                    if let Some(b) = *batch_size {
                        let chunk_size = b.div_ceil(sparses.len());

                        for (sparse, chunk) in sparses.iter_mut().zip(vals.chunks(nnz * chunk_size)) {
                            unsafe {
                                load_sparse_nonblocking(
                                    sparse.sparse_mut()?,
                                    *shape,
                                    *nnz,
                                    Some(chunk.len() / nnz),
                                    chunk,
                                )?;
                            }
                        }
                    } else {
                        for sparse in sparses {
                            unsafe {
                                load_sparse_nonblocking(sparse.sparse_mut()?, *shape, *nnz, None, vals)?;
                            }
                        }
                    }
                }
                HostMatrix::Dense(HostDenseMatrix { vals, shape, batch_size }) => {
                    let denses = self.inputs.get_mut(id).unwrap();

                    if let Some(b) = *batch_size {
                        let chunk_size = b.div_ceil(denses.len());

                        for (dense, chunk) in denses.iter_mut().zip(vals.chunks(shape.size() * chunk_size)) {
                            unsafe {
                                load_dense_nonblocking(
                                    dense.dense_mut()?,
                                    *shape,
                                    Some(chunk.len() / shape.size()),
                                    chunk,
                                )?;
                            }
                        }
                    } else {
                        for dense in denses {
                            unsafe {
                                load_dense_nonblocking(dense.dense_mut()?, *shape, None, vals)?;
                            }
                        }
                    }
                }
            }
        }

        for d in &self.devices {
            d.synchronise()?;
        }

        Ok(())
    }

    pub fn load_into_graph<G: GraphLike<D>>(&mut self, graph: &mut G) -> Result<(), TrainerError<D>> {
        for (id, matrices) in &mut self.inputs {
            if let Some(idx) = graph.primary().input_idx(id) {
                let tensors =
                    graph.get_all(GraphNodeId::new(idx, GraphNodeIdTy::Values)).map_err(TrainerError::Unexpected)?;

                for (tensor, matrix) in tensors.into_iter().zip(matrices.iter_mut()) {
                    tensor.swap_with(matrix).map_err(TrainerError::Unexpected)?;
                }
            }
        }

        Ok(())
    }

    pub fn copy_into_graph<G: GraphLike<D>>(&self, graph: &mut G) -> Result<(), TrainerError<D>> {
        for (id, matrices) in &self.inputs {
            if let Some(idx) = graph.primary().input_idx(id) {
                let tensors =
                    graph.get_all(GraphNodeId::new(idx, GraphNodeIdTy::Values)).map_err(TrainerError::Unexpected)?;

                for (tensor, matrix) in tensors.into_iter().zip(matrices.iter()) {
                    matrix.copy_into(&mut tensor.borrow_mut().values).map_err(TrainerError::Unexpected)?;
                }
            }
        }

        Ok(())
    }
}
