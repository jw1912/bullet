use std::num::NonZeroUsize;

use acyclib::graph::NodeId;

use crate::{
    device::{Device, blas::GemmConfig},
    graph::{
        GraphFunction, GraphNodeId, GraphNodeIdTy,
        instruction::{self, MatmulType},
        ir::{
            BackendMarker, GraphIR, GraphIRError,
            node::AnnotatedNode,
            operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, unary::Reduce, util},
            shape::Shape,
        },
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

#[derive(Clone, Debug)]
pub struct Matmul {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
    pub transa: bool,
    pub transb: bool,
}

impl<B: BackendMarker> GraphIROperationBase<B> for Matmul {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.a, self.b]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.a, true)?;
        util::check_dense_eq(ir, &self.b, true)?;
        util::check_matmul(self.a.shape.maybe_transpose(self.transa), self.b.shape.maybe_transpose(self.transb))
            .map_err(GraphIRError::Op)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Matmul
where
    B::Backend: Device,
{
    fn forward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);
        let bsn = util::batch_size_node(ir, &[self.a, self.b]);

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input: GraphNodeId::new(bsn, GraphNodeIdTy::Values), output });

        let ty = matmul_ty(ir.get(self.a.idx).unwrap().ty().batched, ir.get(self.b.idx).unwrap().ty().batched);

        func.push(instruction::Matmul {
            cfg: GemmConfig::new(1.0, 0.0, self.a.shape, self.transa, self.b.shape, self.transb),
            input_a: GraphNodeId::new(self.a.idx, GraphNodeIdTy::Values),
            input_b: GraphNodeId::new(self.b.idx, GraphNodeIdTy::Values),
            output,
            ty,
        });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let mut func = GraphFunction::default();

        let shape_o = self.a.shape.maybe_transpose(self.transa) * self.b.shape.maybe_transpose(self.transb);

        let ty = matmul_ty(ir.get(self.a.idx).unwrap().ty().batched, ir.get(self.b.idx).unwrap().ty().batched);

        if ir.get(self.a.idx).unwrap().ty().requires_grad {
            let output = GraphNodeId::new(self.a.idx, GraphNodeIdTy::Gradients);
            let b = GraphNodeId::new(self.b.idx, GraphNodeIdTy::Values);
            let o = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);
            let ty = match ty {
                MatmulType::BatBat | MatmulType::NobNob => ty,
                MatmulType::NobBat => MatmulType::BatBatRed,
                MatmulType::BatBatRed => unimplemented!(),
            };

            func.push(instruction::MaybeUpdateBatchSize {
                input: GraphNodeId::new(self.a.idx, GraphNodeIdTy::Values),
                output,
            });

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

        if ir.get(self.b.idx).unwrap().ty().requires_grad {
            let output = GraphNodeId::new(self.b.idx, GraphNodeIdTy::Gradients);
            let a = GraphNodeId::new(self.a.idx, GraphNodeIdTy::Values);
            let o = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize {
                input: GraphNodeId::new(self.b.idx, GraphNodeIdTy::Values),
                output,
            });

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

#[derive(Clone, Debug)]
pub struct Affine {
    pub weights: AnnotatedNode,
    pub biases: AnnotatedNode,
    pub inputs: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperationBase<B> for Affine {
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

    fn ancillary_buffers(&self, ir: &GraphIR<B>) -> Result<Vec<(Shape, Option<NonZeroUsize>)>, GraphIRError> {
        let matmul = Matmul { a: self.weights, b: self.inputs, transa: false, transb: false };
        matmul.ancillary_buffers(ir)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Affine {
    fn forward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let matmul = Matmul { a: self.weights, b: self.inputs, transa: false, transb: false };

        let mut func = <Matmul as GraphIROperationCompilable<B>>::forward_pass(&matmul, ir, output_node);

        let input = GraphNodeId::new(self.biases.idx, GraphNodeIdTy::Values);
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);

        if !ir.get(output_node).unwrap().ty().batched {
            func.push(instruction::LinearCombination { input_mul: 1.0, output_mul: 1.0, input, output });
        } else {
            func.push(instruction::LinearCombinationSplat { input_mul: 1.0, output_mul: 1.0, input, output });
        }

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let matmul = Matmul { a: self.weights, b: self.inputs, transa: false, transb: false };

        let mut func = <Matmul as GraphIROperationCompilable<B>>::backward_pass(&matmul, ir, output_node);

        let info = ir.get(self.biases.idx).unwrap().ty();

        if info.requires_grad {
            let input = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);
            let output = GraphNodeId::new(self.biases.idx, GraphNodeIdTy::Gradients);
            let values = GraphNodeId::new(self.biases.idx, GraphNodeIdTy::Values);

            let input_mul = 1.0;
            let output_mul = 1.0;
            let reduction = Reduce::Sum;

            func.push(instruction::MaybeUpdateBatchSize { input: values, output });

            if info.batched || !ir.get(output_node).unwrap().ty().batched {
                func.push(instruction::LinearCombination { input, output, input_mul, output_mul });
            } else {
                func.push(instruction::ReduceAcrossBatch { input, output, input_mul, output_mul, reduction });
            }
        }

        func
    }
}
