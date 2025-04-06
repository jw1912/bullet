use bullet_core::{
    backend::device::{DeviceBuffer, OperationError},
    graph::ir::{op::DiffableFromOutput, shape::Shape},
};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::CudaBuffer;

use super::CudaError;

pub type OperationResult = Result<(), OperationError<CudaError>>;

fn activation_str(activation: DiffableFromOutput) -> &'static str {
    match activation {
        DiffableFromOutput::Identity => "",
        DiffableFromOutput::ReLU => "Relu",
        DiffableFromOutput::CReLU => "Crelu",
        DiffableFromOutput::SCReLU => "Screlu",
        DiffableFromOutput::SqrReLU => "SqrRelu",
        DiffableFromOutput::Sigmoid => "Sigmoid",
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sparse_affine(
    batch_size: usize,
    stride: Option<bool>,
    activation: DiffableFromOutput,
    input_a: &CudaBuffer<f32>,
    shape_a: Shape,
    input_b: &CudaBuffer<i32>,
    shape_b: Shape,
    nnz: usize,
    input_c: Option<&CudaBuffer<f32>>,
    input_c_batched: bool,
    output: &mut CudaBuffer<f32>,
) -> OperationResult {
    let shape_o = shape_a * shape_b;

    let (stride, offset) = if let Some(b) = stride { (2, if b { shape_a.rows() } else { 0 }) } else { (1, 0) };

    if shape_a.size() > input_a.size()
        || batch_size * nnz > input_b.size()
        || batch_size * shape_o.size() * stride > output.size()
    {
        return Err(OperationError::IndexOutOfBounds);
    }

    let m = shape_a.rows() as u32;
    let k = batch_size as u32;

    let align = m % 4 == 0 && m >= 128;

    let act = activation_str(activation);

    const MAXIMUM_BLOCKS_Y: u32 = 32768;

    let (chunks, threads, smem, align_str) = if align {
        let m4 = m / 4;
        let threads = m4.min(1024);
        let chunks = m4.div_ceil(threads);
        (chunks, threads, 4 * nnz as u32, "Aligned")
    } else {
        let threads = m.min(1024);
        let chunks = m.div_ceil(threads);
        (chunks, threads, 0, "")
    };

    let ky = k.min(MAXIMUM_BLOCKS_Y);
    let kz = k.div_ceil(MAXIMUM_BLOCKS_Y);
    let grid_dim = (chunks, ky, kz);
    let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: smem };

    if let Some(c) = input_c {
        if shape_o.size() * if input_c_batched { batch_size } else { 1 } > c.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        let func_name = format!("SparseAffine{align_str}Fwd{act}");
        let func = output.device.module.load_function(&func_name).map_err(CudaError::Driver)?;

        unsafe {
            output
                .device
                .stream
                .launch_builder(&func)
                .arg(&(stride as i32))
                .arg(&(nnz as i32))
                .arg(&(m as i32))
                .arg(&(k as i32))
                .arg(&input_c_batched)
                .arg(&input_a.buf)
                .arg(&input_b.buf)
                .arg(&c.buf)
                .arg(&mut output.buf.slice_mut(offset..))
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    } else {
        let func_name = format!("SparseMatmul{align_str}Fwd{act}");
        let func = output.device.module.load_function(&func_name).map_err(CudaError::Driver)?;

        unsafe {
            output
                .device
                .stream
                .launch_builder(&func)
                .arg(&(stride as i32))
                .arg(&(nnz as i32))
                .arg(&(m as i32))
                .arg(&(k as i32))
                .arg(&input_a.buf)
                .arg(&input_b.buf)
                .arg(&mut output.buf.slice_mut(offset..))
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn backprop_sparse_affine(
    batch_size: usize,
    stride: Option<bool>,
    activation: DiffableFromOutput,
    input_a: &CudaBuffer<f32>,
    input_a_grad: &mut CudaBuffer<f32>,
    shape_a: Shape,
    input_b: &CudaBuffer<i32>,
    shape_b: Shape,
    nnz: usize,
    _input_c: Option<&CudaBuffer<f32>>,
    input_c_grad: Option<&mut CudaBuffer<f32>>,
    input_c_batched: bool,
    outputs: &CudaBuffer<f32>,
    output_grad: &CudaBuffer<f32>,
) -> OperationResult {
    let shape_o = shape_a * shape_b;

    let (stride, offset) = if let Some(b) = stride { (2, if b { shape_a.rows() } else { 0 }) } else { (1, 0) };

    assert_eq!(shape_b.cols(), 1);
    assert_eq!(shape_o.cols(), 1);
    if shape_a.size() > input_a.size()
        || shape_a.size() > input_a_grad.size()
        || batch_size * nnz > input_b.size()
        || batch_size * shape_o.size() > outputs.size()
        || batch_size * shape_o.size() * stride > output_grad.size()
    {
        return Err(OperationError::IndexOutOfBounds);
    }

    let m = shape_a.rows() as u32;
    let k = batch_size as u32;

    let act = activation_str(activation);

    const MAXIMUM_BLOCKS_Y: u32 = 32768;
    let threads = m.min(1024);
    let chunks = m.div_ceil(threads);
    let ky = k.min(MAXIMUM_BLOCKS_Y);
    let kz = k.div_ceil(MAXIMUM_BLOCKS_Y);
    let grid_dim = (chunks, ky, kz);
    let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

    if let Some(c_grad) = input_c_grad {
        if shape_o.size() * if input_c_batched { batch_size } else { 1 } > c_grad.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        let func_name = format!("SparseAffineBwd{act}");
        let func = input_a.device.module.load_function(&func_name).map_err(CudaError::Driver)?;

        unsafe {
            input_a
                .device
                .stream
                .launch_builder(&func)
                .arg(&(stride as i32))
                .arg(&(nnz as i32))
                .arg(&(m as i32))
                .arg(&(k as i32))
                .arg(&input_c_batched)
                .arg(&input_b.buf)
                .arg(&outputs.buf.slice(offset..))
                .arg(&output_grad.buf.slice(offset..))
                .arg(&mut input_a_grad.buf)
                .arg(&mut c_grad.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    } else {
        let func_name = format!("SparseMatmulBwd{act}");
        let func = input_a.device.module.load_function(&func_name).map_err(CudaError::Driver)?;

        unsafe {
            input_a
                .device
                .stream
                .launch_builder(&func)
                .arg(&(stride as i32))
                .arg(&(nnz as i32))
                .arg(&(m as i32))
                .arg(&(k as i32))
                .arg(&input_b.buf)
                .arg(&outputs.buf.slice(offset..))
                .arg(&output_grad.buf.slice(offset..))
                .arg(&mut input_a_grad.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    };

    Ok(())
}
