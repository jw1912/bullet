use std::num::NonZeroUsize;

use acyclib::{
    dag::NodeId,
    device::{
        function::{DeviceFunction, MaybeUpdateBatchSize, Set},
        operation::DiffableFromOutput,
        tensor::Shape,
    },
    graph::{
        Graph, GraphNodeIdTy,
        ir::{
            GraphIR, GraphIRError, GraphIRMethods,
            node::AnnotatedNode,
            operation::{
                GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, affine::Matmul,
                sparse::SparseAffineActivate, util,
            },
            passes::{GraphIRSimplePass, downcast},
        },
    },
};

use crate::{
    CudaDevice, CudaMarker,
    kernel::{Expr, Kernel, KernelArgs, KernelInput},
};

#[derive(Clone, Copy, Debug)]
pub struct FuseSparseAffineActivateWithMatmul;

impl GraphIRSimplePass<CudaMarker> for FuseSparseAffineActivateWithMatmul {
    fn try_pass_on_node(&self, ir: &mut GraphIR<CudaMarker>, target: NodeId) -> Result<bool, GraphIRError> {
        let op = ir.get(target)?.op();

        if let Some(Matmul { a, b, transa: false, transb: false }) = downcast(op) {
            let valid = |x: AnnotatedNode| {
                let data = ir.get(x.idx)?.ty();
                Ok::<bool, GraphIRError>(data.requires_grad && !data.batched && data.sparse.is_none())
            };

            let parent = ir.get(b.idx)?;

            if parent.children() == 1 && valid(a)? && a.shape.rows() == 1 {
                if let Some(SparseAffineActivate { weights, biases: Some(biases), indices, values: None, activation }) =
                    downcast(parent.op())
                {
                    let hl = weights.shape.rows();
                    if hl % 128 == 0 && (128..=4096).contains(&hl) && valid(weights)? && valid(biases)? {
                        ir.replace(
                            target,
                            SparseAffineUnaryMatmul { weights, biases, indices, out_weights: a, activation },
                        )?;
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SparseAffineUnaryMatmul {
    pub weights: AnnotatedNode,
    pub biases: AnnotatedNode,
    pub indices: AnnotatedNode,
    pub out_weights: AnnotatedNode,
    pub activation: DiffableFromOutput,
}

impl GraphIROperationBase<CudaMarker> for SparseAffineUnaryMatmul {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.weights, self.biases, self.indices, self.out_weights]
    }

    fn output_shape(&self, ir: &GraphIR<CudaMarker>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.indices, false)?;
        util::check_dense_eq(ir, &self.weights, true)?;
        util::check_dense_eq(ir, &self.biases, true)?;
        util::check_dense_eq(ir, &self.out_weights, true)?;
        util::check_not_batched(ir, &self.weights)?;
        util::check_not_batched(ir, &self.biases)?;
        util::check_not_batched(ir, &self.out_weights)?;
        util::check_no_grad(ir, &[&self.indices])?;
        util::check_has_grad(ir, &[&self.weights, &self.biases, &self.out_weights])?;

        let hl_shape = util::check_matmul(self.weights.shape, self.indices.shape)?;
        let out_shape = util::check_matmul(self.out_weights.shape, hl_shape)?;

        if hl_shape.rows() % 128 != 0 || hl_shape.rows() < 128 || hl_shape.rows() > 4096 {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.weights.shape)));
        }

        if hl_shape.cols() != 1 {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.indices.shape)));
        }

        if out_shape != Shape::new(1, 1) {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.out_weights.shape)));
        }

        Ok(out_shape)
    }

    fn ancillary_buffers(
        &self,
        _ir: &GraphIR<CudaMarker>,
    ) -> Result<Vec<(Shape, Option<NonZeroUsize>, bool)>, GraphIRError> {
        let hl_shape = util::check_matmul(self.weights.shape, self.indices.shape)?;
        Ok(vec![(hl_shape, None, true)])
    }

    fn shorthand(&self) -> String {
        "SparseAffineUnaryMatmul".to_string()
    }
}

const MAXIMUM_BLOCKS_Y: i32 = 32768;

impl GraphIROperationCompilable<CudaMarker> for SparseAffineUnaryMatmul {
    fn forward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let mut func = DeviceFunction::default();

        let weights = graph.get_ref(self.weights.idx, GraphNodeIdTy::Values);
        let biases = graph.get_ref(self.biases.idx, GraphNodeIdTy::Values);
        let out_weights = graph.get_ref(self.out_weights.idx, GraphNodeIdTy::Values);

        let indices = graph.get_ref(self.indices.idx, GraphNodeIdTy::Values);
        let borrow = indices.sparse();
        let nnz = borrow.nnz();
        let batched = borrow.batch_size().is_some();
        drop(borrow);

        let hl = graph.get_ref(output_node, GraphNodeIdTy::Ancillary(0));
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        func.push(MaybeUpdateBatchSize { input: indices.clone(), output: hl.clone() });
        func.push(MaybeUpdateBatchSize { input: indices.clone(), output: output.clone() });
        func.push(Set { id: output.clone(), val: 0.0 });

        let layout = None;
        let mutable = false;

        let inputs = vec![
            KernelInput::Size(Expr::Var),
            KernelInput::Slice { slice: weights, layout, mutable, batched: false, shape: self.weights.shape },
            KernelInput::Slice { slice: biases, layout, mutable, batched: false, shape: self.biases.shape },
            KernelInput::Slice { slice: out_weights, layout, mutable, batched: false, shape: self.out_weights.shape },
            KernelInput::Slice { slice: indices, layout: Some(nnz), mutable, batched, shape: self.indices.shape },
            KernelInput::Slice { shape: hl.shape(), slice: hl, layout, mutable: true, batched },
            KernelInput::Slice { shape: Shape::new(1, 1), slice: output, layout, mutable: true, batched },
        ];

        let m = self.weights.shape.rows();
        assert_eq!(m % 128, 0);
        assert!((128..=4096).contains(&m));

        let m4 = m / 4;
        let threads = m4.min(1024);

        let maxy = Expr::Const(MAXIMUM_BLOCKS_Y);
        let batch_size = Expr::Var;
        let ky = batch_size.min(&maxy);
        let kx = (batch_size + maxy.clone() - 1) / maxy;
        let grid_dim = [kx, ky, Expr::Const(1)];
        let block_dim = [Expr::Const(threads as i32), Expr::Const(1), Expr::Const(1)];

        let args = KernelArgs { inputs, grid_dim, block_dim, shared_mem_bytes: Expr::Const(0) };

        let code = include_str!("sl/fwd.cu")
            .lines()
            .skip(7)
            .map(|x| format!("{x}\n"))
            .collect::<String>()
            .replace(
                "ACT_FN",
                match self.activation {
                    DiffableFromOutput::Identity => "x",
                    DiffableFromOutput::ReLU => "max(x, 0.0F)",
                    DiffableFromOutput::CReLU => "min(max(x, 0.0F), 1.0F)",
                    DiffableFromOutput::SCReLU => "min(max(x, 0.0F), 1.0F) * min(max(x, 0.0F), 1.0F)",
                    DiffableFromOutput::SqrReLU => "max(x, 0.0F) * max(x, 0.0F)",
                    DiffableFromOutput::Sigmoid => "1.0F / (1.0F + expf(-x))",
                },
            )
            .replace("DECL_MAXY", &MAXIMUM_BLOCKS_Y.to_string())
            .replace("DECL_M", &m.to_string())
            .replace("DECL_NNZ", &nnz.to_string());

        let kernel = unsafe { Kernel::new("SparseAffineUnaryMatmul".to_string(), code, args) };

        func.push(kernel.unwrap());

        func
    }

    fn backward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let mut func = DeviceFunction::default();

        let out_weights = graph.get_ref(self.out_weights.idx, GraphNodeIdTy::Values);
        let hl = graph.get_ref(output_node, GraphNodeIdTy::Ancillary(0));
        let indices = graph.get_ref(self.indices.idx, GraphNodeIdTy::Values);
        let output_grad = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

        let weights_grad = graph.get_ref(self.weights.idx, GraphNodeIdTy::Gradients);
        let biases_grad = graph.get_ref(self.biases.idx, GraphNodeIdTy::Gradients);
        let out_weights_grad = graph.get_ref(self.out_weights.idx, GraphNodeIdTy::Gradients);

        let borrow = indices.sparse();
        let nnz = borrow.nnz();
        let batched = borrow.batch_size().is_some();
        drop(borrow);

        let m = self.weights.shape.rows();
        assert_eq!(m % 128, 0);
        assert!((128..=4096).contains(&m));

        let batch_size = Expr::Var;
        let maxy = Expr::Const(MAXIMUM_BLOCKS_Y);
        let threads = m.min(1024);
        let chunks = m.div_ceil(threads);
        let ky = batch_size.min(&maxy);
        let kz = (batch_size.clone() + maxy.clone() - 1) / maxy;
        let grid_dim = [Expr::Const(chunks as i32), ky, kz];
        let block_dim = [Expr::Const(threads as i32), Expr::Const(1), Expr::Const(1)];
        let shared_mem_bytes = Expr::Const(0);

        let layout = None;
        let mutable = false;

        let inputs = vec![
            KernelInput::Size(batch_size),
            KernelInput::Slice { slice: indices, layout: Some(nnz), mutable, batched, shape: self.indices.shape },
            KernelInput::Slice { shape: hl.shape(), slice: hl, layout, mutable, batched },
            KernelInput::Slice { slice: output_grad, layout, mutable, batched, shape: Shape::new(1, 1) },
            KernelInput::Slice { slice: out_weights, layout, mutable, batched: false, shape: self.out_weights.shape },
            KernelInput::Slice {
                slice: out_weights_grad,
                layout,
                mutable: true,
                batched: false,
                shape: self.out_weights.shape,
            },
            KernelInput::Slice {
                slice: weights_grad,
                layout,
                mutable: true,
                batched: false,
                shape: self.weights.shape,
            },
            KernelInput::Slice { slice: biases_grad, layout, mutable: true, batched: false, shape: self.biases.shape },
        ];

        let args = KernelArgs { inputs, grid_dim, block_dim, shared_mem_bytes };

        let code = include_str!("sl/bwd.cu")
            .lines()
            .skip(7)
            .map(|x| format!("{x}\n"))
            .collect::<String>()
            .replace(
                "INV_DERIV",
                match self.activation {
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
            .replace("DECL_NNZ", &nnz.to_string());

        let kernel = unsafe { Kernel::new("SparseAffineUnaryMatmulBackward".to_string(), code, args) };

        func.push(kernel.unwrap());

        func
    }
}
