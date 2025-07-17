use bullet_core::{
    device::{DeviceBuffer, OperationError},
    graph::ir::{operation::unary::DiffableFromOutput, shape::Shape},
};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::CudaBuffer;

use super::CudaError;

const MAXIMUM_BLOCKS_Y: u32 = 32768;

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
) -> Result<(), OperationError<CudaError>> {
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

    let vectorise = m % 4 == 0 && m >= 128;

    let (chunks, threads, smem) = if vectorise {
        let m4 = m / 4;
        let threads = m4.min(1024);
        let chunks = m4.div_ceil(threads);
        (chunks, threads, 4 * nnz as u32)
    } else {
        let threads = m.min(1024);
        let chunks = m.div_ceil(threads);
        (chunks, threads, 0)
    };

    let ky = k.min(MAXIMUM_BLOCKS_Y);
    let kz = k.div_ceil(MAXIMUM_BLOCKS_Y);
    let grid_dim = (chunks, ky, kz);
    let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: smem };

    let kernel_cfg = KernelConfig { bias: input_c.map(|_| input_c_batched), m, act: activation, nnz: nnz as u32 };

    let name = kernel_cfg.name();
    let func = unsafe { output.device.get_custom_func_or_rtc(&name, || kernel(kernel_cfg, vectorise))? };

    let mut builder = output.device.stream.launch_builder(&func);

    let stride = stride as i32;
    let k = k as i32;
    let mut view = output.buf.slice_mut(offset..);

    let mut builder = builder.arg(&stride).arg(&k).arg(&input_a.buf).arg(&input_b.buf).arg(&mut view);

    if let Some(c) = input_c {
        if shape_o.size() * if input_c_batched { batch_size } else { 1 } > c.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        builder = builder.arg(&c.buf);
    }

    unsafe {
        builder.launch(cfg).map_err(CudaError::Driver)?;
    }

    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct KernelConfig {
    pub m: u32,
    pub nnz: u32,
    pub act: DiffableFromOutput,
    pub bias: Option<bool>,
}

impl KernelConfig {
    pub fn name(self) -> String {
        let KernelConfig { m, nnz, act, bias } = self;
        let act = match act {
            DiffableFromOutput::Identity => "Identity",
            DiffableFromOutput::ReLU => "Relu",
            DiffableFromOutput::CReLU => "Crelu",
            DiffableFromOutput::SCReLU => "Screlu",
            DiffableFromOutput::SqrReLU => "SqrRelu",
            DiffableFromOutput::Sigmoid => "Sigmoid",
        };

        format!("SparseAffine{act}_{m}_{nnz}_{bias:?}")
    }
}

fn act_str(act: DiffableFromOutput) -> &'static str {
    match act {
        DiffableFromOutput::Identity => "x",
        DiffableFromOutput::ReLU => "x > 0.0F ? x : 0.0F",
        DiffableFromOutput::CReLU => "x < 0.0F ? 0.0F : (x > 1.0F ? 1.0F : x)",
        DiffableFromOutput::SCReLU => "x < 0.0F ? 0.0F : (x > 1.0F ? 1.0F : (x * x))",
        DiffableFromOutput::SqrReLU => "x < 0.0F ? 0.0F : (x * x)",
        DiffableFromOutput::Sigmoid => "1.0F / (1.0F + expf(-x))",
    }
}

fn kernel(cfg: KernelConfig, vectorise: bool) -> String {
    let KernelConfig { m, nnz, act, bias } = cfg;

    let op = format!("__device__ float op(float x) {{ return {}; }}", act_str(act));

    let code = if vectorise { vectorised_kernel(bias) } else { fallback_kernel(bias) };

    let bias_args = if bias.is_some() { ", const float* B" } else { "" };

    format!(
        "
        constexpr int MaximumBlocksY = {MAXIMUM_BLOCKS_Y};

        {op}

        extern \"C\" __global__ void kernel(
            const int stride,
            const int k,
            const float* A,
            const int* X,
            float* Y{bias_args})
        {{
            constexpr int m = {m};
            constexpr int nnz = {nnz};
            const int loc = MaximumBlocksY * blockIdx.z + blockIdx.y;
            const int row = blockIdx.x * blockDim.x + threadIdx.x;
            {code}
        }}"
    )
}

fn vectorised_kernel(bias: Option<bool>) -> String {
    let offset = if bias.unwrap_or(false) { "m4 * loc" } else { "0" };
    let sum = if bias.is_some() {
        "reinterpret_cast<const float4*>(B)[offset + row]"
    } else {
        "make_float4(0.0F, 0.0F, 0.0F, 0.0F)"
    };

    format!(
        "
        extern __shared__ int sX[];

        constexpr int m4 = m / 4;

        if (row >= m4 || loc >= k) return;

        if (threadIdx.x < nnz)
        {{
            for (int i = threadIdx.x; i < nnz; i += blockDim.x)
            {{
                sX[i] = X[nnz * loc + i];
            }}
        }}

        __syncthreads();

        const int offset = {offset};
        float4 val = {sum};

        for (int i = 0; i < nnz; i++) {{
            const int j = sX[i];

            if (j == -1) break;

            const float4 a = reinterpret_cast<const float4*>(A)[j * m4 + row];

            val.x += a.x;
            val.y += a.y;
            val.z += a.z;
            val.w += a.w;
        }}

        val.x = op(val.x);
        val.y = op(val.y);
        val.z = op(val.z);
        val.w = op(val.w);

        reinterpret_cast<float4*>(Y)[stride * m4 * loc + row] = val;"
    )
}

fn fallback_kernel(bias: Option<bool>) -> String {
    let offset = if bias.unwrap_or(false) { "m * loc" } else { "0" };
    let sum = if bias.is_some() { "B[offset + row]" } else { "0.0F" };

    format!(
        "
        if (row >= m || loc >= k) return;

        const int offset = {offset};
        float sum = {sum};

        for (int i = 0; i < nnz; i++) {{
            const int j = X[nnz * loc + i];
            if (j == -1) break;
            sum += A[j * m + row];
        }}

        Y[stride * m * loc + row] = op(sum);"
    )
}
