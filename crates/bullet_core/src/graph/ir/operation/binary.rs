use std::num::NonZeroUsize;

use acyclib::graph::NodeId;

use crate::graph::{
    GraphFunction, GraphNodeId, GraphNodeIdTy, instruction,
    ir::{
        BackendMarker, GraphIR, GraphIRError,
        node::AnnotatedNode,
        operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, util},
        shape::Shape,
    },
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbsPowerError {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
    pub power: f32,
}

impl<B: BackendMarker> GraphIROperationBase<B> for AbsPowerError {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.a, self.b]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.a, true)?;
        util::check_dense_eq(ir, &self.b, true)?;
        util::check_same_batching(ir, &[&self.a, &self.b])?;

        if self.a.shape == self.b.shape {
            Ok(self.a.shape)
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![self.a.shape, self.b.shape])))
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for AbsPowerError {
    fn forward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);
        let bsn = util::batch_size_node(ir, &[self.a, self.b]);

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input: GraphNodeId::new(bsn, GraphNodeIdTy::Values), output });

        func.push(instruction::AbsPowerError {
            a: GraphNodeId::new(self.a.idx, GraphNodeIdTy::Values),
            b: GraphNodeId::new(self.b.idx, GraphNodeIdTy::Values),
            power: self.power,
            output,
        });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let a = GraphNodeId::new(self.a.idx, GraphNodeIdTy::Values);
        let b = GraphNodeId::new(self.b.idx, GraphNodeIdTy::Values);
        let output_grad = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        if ir.get(self.a.idx).unwrap().ty().requires_grad {
            let grd = GraphNodeId::new(self.a.idx, GraphNodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input: a, output: grd });
            func.push(instruction::AbsPowerErrorBackward { a, b, c: output_grad, output: grd, power: self.power });
        }

        if ir.get(self.b.idx).unwrap().ty().requires_grad {
            let grd = GraphNodeId::new(self.b.idx, GraphNodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input: b, output: grd });
            func.push(instruction::AbsPowerErrorBackward {
                a: b,
                b: a,
                c: output_grad,
                output: grd,
                power: self.power,
            });
        }

        func
    }
}

#[derive(Clone, Debug)]
pub struct Concat {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperationBase<B> for Concat {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.a, self.b]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.a, true)?;
        util::check_dense_eq(ir, &self.b, true)?;
        util::check_same_batching(ir, &[&self.a, &self.b])?;

        let ash = self.a.shape;

        if ash.cols() != 1 {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(ash)));
        }

        if ash.cols() == self.b.shape.cols() {
            Ok(Shape::new(ash.rows() + self.b.shape.rows(), ash.cols()))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![ash, self.b.shape])))
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Concat {
    fn forward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);
        let a = GraphNodeId::new(self.a.idx, GraphNodeIdTy::Values);
        let b = GraphNodeId::new(self.b.idx, GraphNodeIdTy::Values);

        let mut func = GraphFunction::default();

        let ainfo = ir.get(self.a.idx).unwrap().ty();

        assert_eq!(ainfo.batched, ir.get(self.b.idx).unwrap().ty().batched);

        func.push(instruction::MaybeUpdateBatchSize { input: a, output });

        func.push(instruction::CopyOrAddStrided {
            input: a,
            output,
            input_offset: 0,
            output_offset: 0,
            add: false,
            len_is_out: false,
        });

        func.push(instruction::CopyOrAddStrided {
            input: b,
            output,
            input_offset: 0,
            output_offset: ainfo.shape.size(),
            add: false,
            len_is_out: false,
        });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let input = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);

        let ainfo = ir.get(self.a.idx).unwrap().ty();
        let binfo = ir.get(self.b.idx).unwrap().ty();

        let mut func = GraphFunction::default();

        if ainfo.requires_grad {
            let output = GraphNodeId::new(self.a.idx, GraphNodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output });

            func.push(instruction::CopyOrAddStrided {
                input,
                output,
                input_offset: 0,
                output_offset: 0,
                add: true,
                len_is_out: true,
            });
        }

        if binfo.requires_grad {
            let output = GraphNodeId::new(self.b.idx, GraphNodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output });

            func.push(instruction::CopyOrAddStrided {
                input,
                output,
                input_offset: ainfo.shape.size(),
                output_offset: 0,
                add: true,
                len_is_out: true,
            });
        }

        func
    }
}

#[derive(Debug)]
pub struct FusedPairwiseMulConcat {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperationBase<B> for FusedPairwiseMulConcat {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.a, self.b]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.a, true)?;
        util::check_dense_eq(ir, &self.b, true)?;
        util::check_same_batching(ir, &[&self.a, &self.b])?;

        let ash = self.a.shape;

        if ash.cols() != 1 {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(ash)));
        }

        if ash.cols() == self.b.shape.cols() {
            Ok(Shape::new(ash.rows() / 2 + self.b.shape.rows() / 2, ash.cols()))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![ash, self.b.shape])))
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for FusedPairwiseMulConcat {
    fn forward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);
        let a = GraphNodeId::new(self.a.idx, GraphNodeIdTy::Values);
        let b = GraphNodeId::new(self.b.idx, GraphNodeIdTy::Values);

        let mut func = GraphFunction::default();

        let ainfo = ir.get(self.a.idx).unwrap().ty();

        assert_eq!(ainfo.batched, ir.get(self.b.idx).unwrap().ty().batched);

        func.push(instruction::MaybeUpdateBatchSize { input: a, output });

        func.push(instruction::PairwiseMul { offset: 0, input: a, output });

        func.push(instruction::PairwiseMul { offset: ainfo.shape.size() / 2, input: b, output });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let input = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);

        let ainfo = ir.get(self.a.idx).unwrap().ty();
        let binfo = ir.get(self.b.idx).unwrap().ty();

        let mut func = GraphFunction::default();

        if ainfo.requires_grad {
            let output = GraphNodeId::new(self.a.idx, GraphNodeIdTy::Gradients);
            let values = GraphNodeId::new(self.a.idx, GraphNodeIdTy::Values);

            func.push(instruction::MaybeUpdateBatchSize { input, output });

            func.push(instruction::PairwiseMulBackward { offset: 0, input, values, output });
        }

        if binfo.requires_grad {
            let output = GraphNodeId::new(self.b.idx, GraphNodeIdTy::Gradients);
            let values = GraphNodeId::new(self.b.idx, GraphNodeIdTy::Values);

            func.push(instruction::MaybeUpdateBatchSize { input, output });

            func.push(instruction::PairwiseMulBackward { offset: ainfo.shape.size() / 2, input, values, output });
        }

        func
    }
}

#[derive(Clone, Debug)]
pub struct Select {
    pub input: AnnotatedNode,
    pub buckets: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperationBase<B> for Select {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input, self.buckets]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        util::check_dense_eq(ir, &self.buckets, false)?;

        if util::check_not_batched(ir, &self.buckets).is_ok() {
            util::check_not_batched(ir, &self.input)?;
        }

        let is = self.input.shape;
        let bs = self.buckets.shape;

        if is.cols() == bs.cols() && is.rows() % bs.rows() == 0 {
            Ok(Shape::new(is.rows() / bs.rows(), is.cols()))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![is, bs])))
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Select {
    fn forward_pass(&self, _ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let input = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Values);
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);
        let buckets = GraphNodeId::new(self.buckets.idx, GraphNodeIdTy::Values);

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input: buckets, output });
        func.push(instruction::Select { input, output, buckets });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        assert!(!ir.get(self.buckets.idx).unwrap().ty().requires_grad);

        let mut func = GraphFunction::default();

        if ir.get(self.input.idx).unwrap().ty().requires_grad {
            let input = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);
            let output = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Gradients);
            let buckets = GraphNodeId::new(self.buckets.idx, GraphNodeIdTy::Values);
            let values = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Values);

            func.push(instruction::MaybeUpdateBatchSize { input: values, output });
            func.push(instruction::SelectBackprop { input, output, buckets });
        }

        func
    }
}

#[derive(Debug)]
pub struct SoftmaxCrossEntropy {
    pub logits: AnnotatedNode,
    pub targets: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperationBase<B> for SoftmaxCrossEntropy {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.logits, self.targets]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.logits, true)?;
        util::check_dense_eq(ir, &self.targets, true)?;
        util::check_same_batching(ir, &[&self.logits, &self.targets])?;
        util::check_no_grad(ir, &[&self.targets])?;

        let shape = self.logits.shape;

        if shape != self.targets.shape {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![shape, self.targets.shape])))
        } else {
            Ok(shape)
        }
    }

    fn ancillary_buffers(&self, _ir: &GraphIR<B>) -> Result<Vec<(Shape, Option<NonZeroUsize>)>, GraphIRError> {
        Ok(vec![(self.logits.shape, None)])
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for SoftmaxCrossEntropy {
    fn forward_pass(&self, _ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<<B as BackendMarker>::Backend> {
        let logits = GraphNodeId::new(self.logits.idx, GraphNodeIdTy::Values);
        let targets = GraphNodeId::new(self.targets.idx, GraphNodeIdTy::Values);
        let smax = GraphNodeId::new(output_node, GraphNodeIdTy::Ancillary(0));
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input: logits, output: smax });
        func.push(instruction::MaybeUpdateBatchSize { input: logits, output });

        func.push(instruction::Softmax { input: logits, output: smax });
        func.push(instruction::CrossEntropy { a: smax, b: targets, output });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<<B as BackendMarker>::Backend> {
        assert!(!ir.get(self.targets.idx).unwrap().ty().requires_grad);

        let mut func = GraphFunction::default();

        if ir.get(self.logits.idx).unwrap().ty().requires_grad {
            let softmax = GraphNodeId::new(output_node, GraphNodeIdTy::Ancillary(0));
            let output_grads = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);
            let targets = GraphNodeId::new(self.targets.idx, GraphNodeIdTy::Values);
            let output = GraphNodeId::new(self.logits.idx, GraphNodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input: softmax, output });
            func.push(instruction::SoftmaxCrossEntropyBackward { softmax, output_grads, targets, output });
        }

        func
    }
}
