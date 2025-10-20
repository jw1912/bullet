use acyclib::{
    dag::NodeId,
    device::function::{self, DeviceFunction},
    graph::{
        Graph, GraphNodeIdTy,
        ir::operation::{GraphIROperationCompilable, unary::FauxQuantise},
    },
};

use crate::{
    CudaDevice, CudaMarker,
    kernel::{Expr, Kernel, KernelArgs, KernelInput},
};

impl GraphIROperationCompilable<CudaMarker> for FauxQuantise {
    fn forward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });

        let op = if self.round { "roundf" } else { "truncf" };
        let q = self.value;
        let code = format!(
            "
            constexpr float Q = static_cast<float>({q});

            extern \"C\" __global__ void kernel(const int size, const float* input, float* output)
            {{
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid < size)
                {{
                    output[tid] = {op}(Q * input[tid]) / Q;
                }}
            }}"
        );

        let threads = Expr::Const(512);
        let size = Expr::Var * input.single_size() as i32;
        let blocks = (size.clone() + threads.clone() - 1) / threads.clone();
        let grid_dim = [blocks, Expr::Const(1), Expr::Const(1)];
        let block_dim = [threads, Expr::Const(1), Expr::Const(1)];

        let layout = None;
        let batched = input.batch_size().is_some();
        let shape = input.shape();
        let inputs = vec![
            KernelInput::Size(size),
            KernelInput::Slice { slice: input, layout, mutable: false, batched, shape },
            KernelInput::Slice { slice: output, layout, mutable: true, batched, shape },
        ];

        let args = KernelArgs { grid_dim, block_dim, shared_mem_bytes: Expr::Const(0), inputs };

        let kernel = unsafe { Kernel::new("FauxQuantise".to_string(), code, args) };

        func.push(kernel.unwrap());

        func
    }

    fn backward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        self.backward::<CudaMarker>(graph, output_node)
    }
}
