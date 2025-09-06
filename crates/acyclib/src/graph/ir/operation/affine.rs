use std::num::NonZeroUsize;

use crate::{
    dag::NodeId,
    device::{
        Device,
        function::{self, DeviceFunction, MatmulType, Reduce},
        operation::GemmConfig,
        tensor::Shape,
    },
    graph::{
        Graph, GraphNodeIdTy,
        ir::{
            BackendMarker, GraphIR, GraphIRError,
            node::AnnotatedNode,
            operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, util},
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
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);
        let bsn = util::batch_size_node::<B>(graph, &[self.a, self.b]);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize {
            input: graph.get_ref(bsn, GraphNodeIdTy::Values),
            output: output.clone(),
        });

        let input_a = graph.get_ref(self.a.idx, GraphNodeIdTy::Values);
        let input_b = graph.get_ref(self.b.idx, GraphNodeIdTy::Values);

        let ty = matmul_ty(input_a.borrow().batch_size().is_some(), input_b.borrow().batch_size().is_some());

        func.push(function::Matmul {
            cfg: GemmConfig::new(1.0, 0.0, self.a.shape, self.transa, self.b.shape, self.transb),
            input_a,
            input_b,
            output,
            ty,
        });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();
        let shape_o = self.a.shape.maybe_transpose(self.transa) * self.b.shape.maybe_transpose(self.transb);

        let input_a = graph.get_ref(self.a.idx, GraphNodeIdTy::Values);
        let input_b = graph.get_ref(self.b.idx, GraphNodeIdTy::Values);

        let ty = matmul_ty(input_a.borrow().batch_size().is_some(), input_b.borrow().batch_size().is_some());

        if let Some(output) = graph.maybe_get_ref(self.a.idx, GraphNodeIdTy::Gradients) {
            let o = graph.get_ref(output_node, GraphNodeIdTy::Gradients);
            let ty = match ty {
                MatmulType::BatBat | MatmulType::NobNob => ty,
                MatmulType::NobBat => MatmulType::BatBatRed,
                MatmulType::BatBatRed => unimplemented!(),
            };

            func.push(function::MaybeUpdateBatchSize { input: input_a.clone(), output: output.clone() });

            let instr = if self.transa {
                function::Matmul {
                    cfg: GemmConfig::new(1.0, 1.0, self.b.shape, self.transb, shape_o, true),
                    output,
                    input_a: input_b.clone(),
                    input_b: o,
                    ty,
                }
            } else {
                function::Matmul {
                    cfg: GemmConfig::new(1.0, 1.0, shape_o, false, self.b.shape, !self.transb),
                    output,
                    input_a: o,
                    input_b: input_b.clone(),
                    ty,
                }
            };

            func.push(instr);
        }

        if let Some(output) = graph.maybe_get_ref(self.b.idx, GraphNodeIdTy::Gradients) {
            let o = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

            func.push(function::MaybeUpdateBatchSize { input: input_b, output: output.clone() });

            if self.transb {
                if ty == MatmulType::NobBat {
                    unimplemented!();
                }

                func.push(function::Matmul {
                    cfg: GemmConfig::new(1.0, 1.0, shape_o, true, self.a.shape, self.transa),
                    output,
                    input_a: o,
                    input_b: input_a,
                    ty,
                });
            } else {
                func.push(function::Matmul {
                    cfg: GemmConfig::new(1.0, 1.0, self.a.shape, !self.transa, shape_o, false),
                    output,
                    input_a,
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

    fn ancillary_buffers(&self, ir: &GraphIR<B>) -> Result<Vec<(Shape, Option<NonZeroUsize>, bool)>, GraphIRError> {
        let matmul = Matmul { a: self.weights, b: self.inputs, transa: false, transb: false };
        matmul.ancillary_buffers(ir)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Affine {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let matmul = Matmul { a: self.weights, b: self.inputs, transa: false, transb: false };

        let mut func = <Matmul as GraphIROperationCompilable<B>>::forward_pass(&matmul, graph, output_node);

        let input = graph.get_ref(self.biases.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        if input.borrow().batch_size().is_some() || output.borrow().batch_size().is_none() {
            func.push(function::LinearCombination { input_mul: 1.0, output_mul: 1.0, input, output });
        } else {
            func.push(function::LinearCombinationSplat { input_mul: 1.0, output_mul: 1.0, input, output });
        }

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let matmul = Matmul { a: self.weights, b: self.inputs, transa: false, transb: false };

        let mut func = <Matmul as GraphIROperationCompilable<B>>::backward_pass(&matmul, graph, output_node);

        if let Some(output) = graph.maybe_get_ref(self.biases.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(output_node, GraphNodeIdTy::Gradients);
            let values = graph.get_ref(self.biases.idx, GraphNodeIdTy::Values);

            let input_mul = 1.0;
            let output_mul = 1.0;
            let reduction = Reduce::Sum;

            func.push(function::MaybeUpdateBatchSize { input: values.clone(), output: output.clone() });

            if values.borrow().batch_size().is_some() || input.borrow().batch_size().is_none() {
                func.push(function::LinearCombination { input, output, input_mul, output_mul });
            } else {
                func.push(function::ReduceAcrossBatch { input, output, input_mul, output_mul, reduction });
            }
        }

        func
    }
}
