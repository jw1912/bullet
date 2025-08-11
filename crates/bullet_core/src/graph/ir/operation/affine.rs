use std::num::NonZeroUsize;

use crate::{
    device::{blas::GemmConfig, Device},
    graph::{
        instruction::{self, MatmulType},
        ir::{
            node::AnnotatedNode,
            operation::{unary::Reduce, util, GraphIROperation, GraphIROperationCompilable, GraphIROperationError},
            shape::Shape,
            BackendMarker, GraphIR, GraphIRError, GraphIRNodeInfo,
        },
        GraphFunction, NodeId, NodeIdTy,
    },
};

fn matmul_ty(batched_a: bool, batched_b: bool) -> MatmulType {
    match (batched_a, batched_b) {
        (true, true) => MatmulType::BatBat,
        (false, false) => MatmulType::NobNob,
        (false, true) => MatmulType::NobBat,
        (true, false) => unimplemented!(),
    }
}

#[derive(Debug)]
pub struct Matmul {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
    pub transa: bool,
    pub transb: bool,
}

impl<B: BackendMarker> GraphIROperation<B> for Matmul {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.a, self.b]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.a, true)?;
        util::check_dense_eq(ir, &self.b, true)?;
        util::check_matmul(self.a.shape.maybe_transpose(self.transa), self.b.shape.maybe_transpose(self.transb))
            .map_err(GraphIRError::Op)
    }

    fn ancillary_buffers(&self, _ir: &GraphIR<B>) -> Result<Vec<(Shape, Option<NonZeroUsize>)>, GraphIRError> {
        let trans_on_bprop = !self.transa && !self.transb && self.a.shape.rows() > 1 && self.a.shape.cols() > 1;
        Ok(if trans_on_bprop { vec![(self.a.shape.transpose(), None)] } else { Vec::new() })
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Matmul
where
    B::Backend: Device,
{
    fn forward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let output = NodeId::new(output_node, NodeIdTy::Values);
        let bsn = util::batch_size_node(node_info, &[self.a, self.b]);

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input: NodeId::new(bsn, NodeIdTy::Values), output });

        let ty = matmul_ty(node_info.get(self.a.idx).unwrap().batched, node_info.get(self.b.idx).unwrap().batched);

        func.push(instruction::Matmul {
            cfg: GemmConfig::new(1.0, 0.0, self.a.shape, self.transa, self.b.shape, self.transb),
            input_a: NodeId::new(self.a.idx, NodeIdTy::Values),
            input_b: NodeId::new(self.b.idx, NodeIdTy::Values),
            output,
            ty,
        });

        func
    }

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let mut func = GraphFunction::default();

        let shape_o = self.a.shape.maybe_transpose(self.transa) * self.b.shape.maybe_transpose(self.transb);

        let ty = matmul_ty(node_info.get(self.a.idx).unwrap().batched, node_info.get(self.b.idx).unwrap().batched);

        if node_info.get(self.a.idx).unwrap().requires_grad {
            let output = NodeId::new(self.a.idx, NodeIdTy::Gradients);
            let b = NodeId::new(self.b.idx, NodeIdTy::Values);
            let o = NodeId::new(output_node, NodeIdTy::Gradients);
            let ty = match ty {
                MatmulType::BatBat | MatmulType::NobNob => ty,
                MatmulType::NobBat => MatmulType::BatBatRed,
                MatmulType::BatBatRed => unimplemented!(),
            };

            func.push(instruction::MaybeUpdateBatchSize { input: NodeId::new(self.a.idx, NodeIdTy::Values), output });

            let instr = if self.transa {
                instruction::Matmul {
                    cfg: GemmConfig::new(1.0, 1.0, self.b.shape, self.transb, shape_o, true),
                    output,
                    input_a: b,
                    input_b: o,
                    ty,
                }
            } else {
                instruction::Matmul {
                    cfg: GemmConfig::new(1.0, 1.0, shape_o, false, self.b.shape, !self.transb),
                    output,
                    input_a: o,
                    input_b: b,
                    ty,
                }
            };

            func.push(instr);
        }

        if node_info.get(self.b.idx).unwrap().requires_grad {
            let output = NodeId::new(self.b.idx, NodeIdTy::Gradients);
            let a = NodeId::new(self.a.idx, NodeIdTy::Values);
            let o = NodeId::new(output_node, NodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input: NodeId::new(self.b.idx, NodeIdTy::Values), output });

            if self.transb {
                if ty == MatmulType::NobBat {
                    unimplemented!();
                }

                func.push(instruction::Matmul {
                    cfg: GemmConfig::new(1.0, 1.0, shape_o, true, self.a.shape, self.transa),
                    output,
                    input_a: o,
                    input_b: a,
                    ty,
                });
            } else if !self.transa && self.a.shape.rows() > 1 && self.a.shape.cols() > 1 && ty == MatmulType::NobBat {
                let a_trans = NodeId::new(output_node, NodeIdTy::Ancillary(0));

                func.push(instruction::Transpose {
                    input: a,
                    output: a_trans,
                    rows: self.a.shape.rows(),
                    cols: self.a.shape.cols(),
                    input_mul: 1.0,
                    output_mul: 0.0,
                });

                func.push(instruction::Matmul {
                    cfg: GemmConfig::new(1.0, 1.0, self.a.shape.transpose(), false, shape_o, false),
                    output,
                    input_a: a_trans,
                    input_b: o,
                    ty,
                });
            } else {
                func.push(instruction::Matmul {
                    cfg: GemmConfig::new(1.0, 1.0, self.a.shape, !self.transa, shape_o, false),
                    output,
                    input_a: a,
                    input_b: o,
                    ty,
                });
            }
        }

        func
    }
}

#[derive(Debug)]
pub struct Affine {
    pub weights: AnnotatedNode,
    pub biases: AnnotatedNode,
    pub inputs: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperation<B> for Affine {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.weights, self.biases, self.inputs]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.weights, true)?;
        util::check_dense_eq(ir, &self.inputs, true)?;
        util::check_dense_eq(ir, &self.biases, true)?;
        util::check_not_batched(ir, &self.weights)?;
        util::check_not_batched(ir, &self.biases)?;

        // N.B:
        // y = A.matmul(x).reshape(b.shape) + b -> mm_shape != b.shape
        // y = A.matmul(x) + b2.reshape(mm_shape) -> mm_shape == b.shape
        let mm_shape = util::check_matmul(self.weights.shape, self.inputs.shape)?;

        if mm_shape.size() == self.biases.shape.size() {
            Ok(self.biases.shape)
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![
                self.weights.shape,
                self.inputs.shape,
            ])))
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Affine {
    fn forward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let matmul = Matmul { a: self.weights, b: self.inputs, transa: false, transb: false };

        let mut func = <Matmul as GraphIROperationCompilable<B>>::forward_pass(&matmul, node_info, output_node);

        let input = NodeId::new(self.biases.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

        if !node_info.get(output_node).unwrap().batched {
            func.push(instruction::LinearCombination { input_mul: 1.0, output_mul: 1.0, input, output });
        } else {
            func.push(instruction::LinearCombinationSplat { input_mul: 1.0, output_mul: 1.0, input, output });
        }

        func
    }

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let matmul = Matmul { a: self.weights, b: self.inputs, transa: false, transb: false };

        let mut func = <Matmul as GraphIROperationCompilable<B>>::backward_pass(&matmul, node_info, output_node);

        let info = node_info.get(self.biases.idx).unwrap();

        if info.requires_grad {
            let input = NodeId::new(output_node, NodeIdTy::Gradients);
            let output = NodeId::new(self.biases.idx, NodeIdTy::Gradients);
            let values = NodeId::new(self.biases.idx, NodeIdTy::Values);

            let input_mul = 1.0;
            let output_mul = 1.0;
            let reduction = Reduce::Sum;

            func.push(instruction::MaybeUpdateBatchSize { input: values, output });

            if info.batched || !node_info.get(output_node).unwrap().batched {
                func.push(instruction::LinearCombination { input, output, input_mul, output_mul });
            } else {
                func.push(instruction::ReduceAcrossBatch { input, output, input_mul, output_mul, reduction });
            }
        }

        func
    }
}
