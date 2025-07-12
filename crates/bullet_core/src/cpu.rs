pub mod base;
pub mod blas;
pub mod cmp;
pub mod sparse;

use std::sync::Arc;

use crate::{
    device::{Device, DeviceBuffer, OperationError, OperationResult},
    graph::ir::{operation::unary::DiffableFromOutput, shape::Shape, BackendMarker},
};

#[derive(Debug, Default)]
pub struct CpuError;

#[derive(Debug, Default)]
pub struct CpuThread;

pub struct CpuBuffer<T> {
    buf: Vec<T>,
    device: Arc<CpuThread>,
}

#[derive(Clone, Copy, Default)]
pub struct CpuMarker;

impl BackendMarker for CpuMarker {
    type Backend = CpuThread;
}

impl<T: Copy + Default> DeviceBuffer<CpuThread, T> for CpuBuffer<T> {
    type BufferError = CpuError;

    fn device(&self) -> Arc<CpuThread> {
        self.device.clone()
    }

    fn new(device: Arc<CpuThread>, size: usize) -> Result<Self, CpuError> {
        Ok(Self { buf: vec![T::default(); size], device })
    }

    fn size(&self) -> usize {
        self.buf.len()
    }

    fn set_zero(&mut self) -> Result<(), CpuError> {
        for elem in &mut self.buf {
            *elem = T::default();
        }

        Ok(())
    }

    fn load_from_device(&mut self, buf: &Self, num: usize) -> Result<(), CpuError> {
        self.buf[..num].copy_from_slice(&buf.buf[..num]);
        Ok(())
    }

    fn load_from_slice(&mut self, buf: &[T]) -> Result<(), CpuError> {
        self.buf[..buf.len()].copy_from_slice(buf);
        Ok(())
    }

    unsafe fn load_non_blocking_from_host(&mut self, buf: &[T]) -> Result<(), Self::BufferError> {
        self.load_from_slice(buf)
    }

    fn write_into_slice(&self, buf: &mut [T], num: usize) -> Result<(), CpuError> {
        buf[..num].copy_from_slice(&self.buf[..num]);
        Ok(())
    }
}

#[allow(unused)]
impl Device for CpuThread {
    type Marker = CpuMarker;

    type BufferF32 = CpuBuffer<f32>;
    type BufferI32 = CpuBuffer<i32>;

    type DeviceError = CpuError;

    type IdType = ();

    fn new(_id: Self::IdType) -> Result<Self, Self::DeviceError> {
        Ok(Self)
    }

    fn synchronise(&self) -> Result<(), Self::DeviceError> {
        Ok(())
    }

    fn get_last_device_error(&self) -> Result<(), Self::DeviceError> {
        Ok(())
    }

    fn sparse_affine_activate(
        batch_size: usize,
        stride: Option<bool>,
        activation: DiffableFromOutput,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        input_b_vals: Option<&Self::BufferF32>,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        input_c_batched: bool,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let shape_o = shape_a * shape_b;

        let (stride, offset) = if let Some(b) = stride { (2, if b { shape_a.rows() } else { 0 }) } else { (1, 0) };

        if shape_a.size() > input_a.size()
            || batch_size * nnz > input_b.size()
            || batch_size * shape_o.size() * stride > output.size()
        {
            return Err(OperationError::IndexOutOfBounds);
        }

        if let Some(c) = input_c {
            if shape_o.size() * if input_c_batched { batch_size } else { 1 } > c.size() {
                return Err(OperationError::IndexOutOfBounds);
            }
        }

        let m = shape_a.rows();
        let k = batch_size;
        let a = &input_a.buf;
        let x = &input_b.buf;
        let v = input_b_vals.map(|v| &*v.buf);
        let b = input_c.map(|c| &*c.buf);
        let bb = input_c_batched;
        let y = &mut output.buf;

        match activation {
            DiffableFromOutput::Identity => sparse::affine_fwd(stride, offset, nnz, m, k, a, x, v, b, bb, y, |x| x),
            DiffableFromOutput::ReLU => {
                sparse::affine_fwd(stride, offset, nnz, m, k, a, x, v, b, bb, y, |x| x.max(0.0))
            }
            DiffableFromOutput::CReLU => {
                sparse::affine_fwd(stride, offset, nnz, m, k, a, x, v, b, bb, y, |x| x.clamp(0.0, 1.0))
            }
            DiffableFromOutput::SCReLU => {
                sparse::affine_fwd(stride, offset, nnz, m, k, a, x, v, b, bb, y, |x| x.clamp(0.0, 1.0).powi(2))
            }
            DiffableFromOutput::SqrReLU => {
                sparse::affine_fwd(stride, offset, nnz, m, k, a, x, v, b, bb, y, |x| x.max(0.0).powi(2))
            }
            DiffableFromOutput::Sigmoid => {
                sparse::affine_fwd(stride, offset, nnz, m, k, a, x, v, b, bb, y, |x| 1.0 / (1.0 + (-x).exp()))
            }
        }

        Ok(())
    }

    fn backprop_sparse_affine_activate(
        batch_size: usize,
        stride: Option<bool>,
        activation: DiffableFromOutput,
        input_a_grad: &mut Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        input_b_vals: Option<&Self::BufferF32>,
        shape_b: Shape,
        nnz: usize,
        input_c_grad: Option<&mut Self::BufferF32>,
        input_c_batched: bool,
        outputs: &Self::BufferF32,
        output_grad: &Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let shape_o = shape_a * shape_b;

        let (stride, offset) = if let Some(b) = stride { (2, if b { shape_a.rows() } else { 0 }) } else { (1, 0) };

        assert_eq!(shape_b.cols(), 1);
        assert_eq!(shape_o.cols(), 1);
        if shape_a.size() > input_a_grad.size()
            || batch_size * nnz > input_b.size()
            || batch_size * shape_o.size() > outputs.size()
            || batch_size * shape_o.size() * stride > output_grad.size()
        {
            return Err(OperationError::IndexOutOfBounds);
        }

        if let Some(ref grad) = input_c_grad {
            if shape_o.size() * if input_c_batched { batch_size } else { 1 } > grad.size() {
                return Err(OperationError::IndexOutOfBounds);
            }
        }

        let m = shape_a.rows();
        let k = batch_size;
        let x = &input_b.buf;
        let v = input_b_vals.map(|v| &*v.buf);
        let y = &outputs.buf;
        let yg = &output_grad.buf;
        let bb = input_c_batched;
        let ag = &mut input_a_grad.buf;
        let bg = input_c_grad.map(|x| &mut *x.buf);

        match activation {
            DiffableFromOutput::Identity => {
                sparse::affine_bwd(stride, offset, nnz, m, k, x, v, y, yg, bb, ag, bg, |x| 1.0)
            }
            DiffableFromOutput::ReLU => {
                sparse::affine_bwd(stride, offset, nnz, m, k, x, v, y, yg, bb, ag, bg, |x| f32::from(x > 0.0))
            }
            DiffableFromOutput::CReLU => sparse::affine_bwd(stride, offset, nnz, m, k, x, v, y, yg, bb, ag, bg, |x| {
                if x > 0.0 && x < 1.0 {
                    1.0
                } else {
                    0.0
                }
            }),
            DiffableFromOutput::SCReLU => sparse::affine_bwd(stride, offset, nnz, m, k, x, v, y, yg, bb, ag, bg, |x| {
                if x > 0.0 && x < 1.0 {
                    2.0 * x.sqrt()
                } else {
                    0.0
                }
            }),
            DiffableFromOutput::SqrReLU => {
                sparse::affine_bwd(stride, offset, nnz, m, k, x, v, y, yg, bb, ag, bg, |x| 2.0 * x.max(0.0).sqrt())
            }
            DiffableFromOutput::Sigmoid => {
                sparse::affine_bwd(stride, offset, nnz, m, k, x, v, y, yg, bb, ag, bg, |x| x * (1.0 - x))
            }
        }

        Ok(())
    }

    fn select(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn select_backprop(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }
}
