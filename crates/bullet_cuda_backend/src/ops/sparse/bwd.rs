use bullet_core::{function, graph::ir::operation::unary::DiffableFromOutput};

use crate::{
    CudaDevice,
    kernel::{Expr, Kernel, KernelArgs, KernelInput},
};

const MAXIMUM_BLOCKS_Y: i32 = 32768;

pub fn kernel(desc: function::BackpropSparseAffineActivate<CudaDevice>) -> Kernel {
    let output_shape = desc.weights_shape * desc.input_shape;
    let indices = desc.indices;

    assert_eq!(desc.weights_shape.size(), desc.weights_grads.shape().size());
    assert_eq!(desc.input_shape.size(), indices.shape().size());
    assert_eq!(desc.input_shape.cols(), 1);
    assert_eq!(output_shape.cols(), 1);

    let bias = desc.biases_grads.as_ref().map(|x| x.batch_size().is_some());

    let batched = indices.batch_size().is_some();
    let nnz = indices.sparse().nnz;
    let m = output_shape.rows();

    let code = kernel_str(bias, nnz, m, desc.activation);

    let batch_size = Expr::Var;

    let mut inputs = vec![
        KernelInput::Size(batch_size.clone()),
        KernelInput::Slice { slice: indices, layout: Some(nnz), mutable: false, batched, shape: desc.input_shape },
        KernelInput::Slice { slice: desc.output, layout: None, mutable: false, batched, shape: output_shape },
        KernelInput::Slice { slice: desc.output_grads, layout: None, mutable: false, batched, shape: output_shape },
        KernelInput::Slice {
            slice: desc.weights_grads,
            layout: None,
            mutable: true,
            batched: false,
            shape: desc.weights_shape,
        },
    ];

    if let Some(bias) = desc.biases_grads {
        let batched = bias.batch_size().is_some();
        let shape = bias.shape();
        assert_eq!(shape.size(), output_shape.size());

        inputs.push(KernelInput::Slice { slice: bias, layout: None, mutable: true, batched, shape: output_shape });
    }

    let maxy = Expr::Const(MAXIMUM_BLOCKS_Y);
    let threads = m.min(1024);
    let chunks = m.div_ceil(threads);
    let ky = batch_size.min(&maxy);
    let kz = (batch_size + maxy.clone() - 1) / maxy;
    let grid_dim = [Expr::Const(chunks as i32), ky, kz];
    let block_dim = [Expr::Const(threads as i32), Expr::Const(1), Expr::Const(1)];

    let shared_mem_bytes = Expr::Const(0);

    let args = KernelArgs { inputs, grid_dim, block_dim, shared_mem_bytes };

    unsafe { Kernel::new("SparseAffineActiveBackward".to_string(), code, args).unwrap() }
}

fn kernel_str(bias: Option<bool>, nnz: usize, m: usize, activation: DiffableFromOutput) -> String {
    include_str!("bwd.cu")
        .lines()
        .skip(8)
        .map(|x| format!("{x}\n"))
        .collect::<String>()
        .replace(
            "INV_DERIV",
            match activation {
                DiffableFromOutput::Identity => "1.0F",
                DiffableFromOutput::ReLU => "x > 0.0F ? 1.0F : 0.0F",
                DiffableFromOutput::CReLU => "x > 0.0F && x < 1.0F ? 1.0F : 0.0F",
                DiffableFromOutput::SCReLU => "x > 0.0F && x < 1.0F ? 2.0F * sqrtf(x) : 0.0F",
                DiffableFromOutput::SqrReLU => "x > 0.0F ? 2.0F * sqrtf(x) : 0.0F",
                DiffableFromOutput::Sigmoid => "x * (1.0F - x)",
            },
        )
        .replace("DECL_MAXY", &MAXIMUM_BLOCKS_Y.to_string())
        .replace("DECL_M", &m.to_string())
        .replace("DECL_NNZ", &nnz.to_string())
        .replace("BIAS_ARG", if bias.is_some() { ",float* Bg" } else { "" })
        .replace(
            "BIAS_BACKPROP",
            match bias {
                None => "",
                Some(true) => {
                    "if (tE != 0.0F) {\
                    \n        const int offset2 = m * loc;\
                    \n        atomicAdd(&Bg[offset2 + row], tE);\
                    \n    }"
                }
                Some(false) => {
                    "if (tE != 0.0F) {\
                    \n        const int offset2 = 0;\
                    \n        atomicAdd(&Bg[offset2 + row], tE);\
                    \n    }"
                }
            },
        )
}
