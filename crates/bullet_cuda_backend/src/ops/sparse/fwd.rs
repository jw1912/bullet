use bullet_core::{function, graph::ir::operation::unary::DiffableFromOutput};

use crate::{
    CudaDevice,
    kernel::{Expr, Kernel, KernelArgs, KernelInput},
};

const MAXIMUM_BLOCKS_Y: u32 = 32768;

pub fn kernel(desc: function::SparseAffineActivate<CudaDevice>) -> Kernel {
    let output_shape = desc.weights_shape * desc.input_shape;
    let indices = desc.indices;

    assert_eq!(desc.weights_shape.size(), desc.weights.shape().size());
    assert_eq!(desc.input_shape.size(), indices.shape().size());
    assert_eq!(desc.input_shape.cols(), 1);
    assert_eq!(output_shape.cols(), 1);

    let bias = desc.biases.as_ref().map(|x| x.batch_size().is_some());

    let batched = indices.batch_size().is_some();
    let nnz = indices.sparse().nnz;
    let m = output_shape.rows();
    let vectorise = m % 4 == 0 && m >= 128;

    let code = kernel_str(bias, nnz, m, desc.activation, vectorise);

    let batch_size = Expr::Var;

    let mut inputs = vec![
        KernelInput::Size(batch_size.clone()),
        KernelInput::Slice {
            slice: desc.weights,
            layout: None,
            mutable: false,
            batched: false,
            shape: desc.weights_shape,
        },
        KernelInput::Slice { slice: indices, layout: Some(nnz), mutable: false, batched, shape: desc.input_shape },
        KernelInput::Slice { slice: desc.output, layout: None, mutable: true, batched, shape: output_shape },
    ];

    if let Some(bias) = desc.biases {
        let batched = bias.batch_size().is_some();
        let shape = bias.shape();
        assert_eq!(shape.size(), output_shape.size());

        inputs.push(KernelInput::Slice { slice: bias, layout: None, mutable: false, batched, shape: output_shape });
    }

    const MAXIMUM_BLOCKS_Y: Expr<i32> = Expr::Const(32768);

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

    let ky = batch_size.min(&MAXIMUM_BLOCKS_Y);
    let kz = (batch_size + MAXIMUM_BLOCKS_Y - 1) / MAXIMUM_BLOCKS_Y;
    let grid_dim = [Expr::Const(chunks as i32), ky, kz];
    let block_dim = [Expr::Const(threads as i32), Expr::Const(1), Expr::Const(1)];
    let shared_mem_bytes = Expr::Const(smem as i32);

    let args = KernelArgs { inputs, grid_dim, block_dim, shared_mem_bytes };

    unsafe { Kernel::new("SparseAffineActiveBackward".to_string(), code, args).unwrap() }
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

fn kernel_str(bias: Option<bool>, nnz: usize, m: usize, activation: DiffableFromOutput, vectorise: bool) -> String {
    let op = format!("__device__ float op(float x) {{ return {}; }}", act_str(activation));

    let code = if vectorise { vectorised_kernel(bias) } else { fallback_kernel(bias) };

    let bias_args = if bias.is_some() { ", const float* B" } else { "" };

    format!(
        "
        constexpr int MaximumBlocksY = {MAXIMUM_BLOCKS_Y};

        {op}

        extern \"C\" __global__ void kernel(
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

        reinterpret_cast<float4*>(Y)[m4 * loc + row] = val;"
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

        Y[m * loc + row] = op(sum);"
    )
}
