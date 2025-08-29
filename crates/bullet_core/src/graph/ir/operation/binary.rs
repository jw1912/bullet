use std::num::NonZeroUsize;

use acyclib::graph::NodeId;

use crate::{
    device::CoreDeviceOps,
    function,
    graph::{
        DeviceFunction, Graph, GraphNodeIdTy,
        ir::{
            BackendMarker, GraphIR, GraphIRError,
            node::AnnotatedNode,
            operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, util},
            shape::Shape,
        },
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
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);
        let bsn = util::batch_size_node::<B>(graph, &[self.a, self.b]);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize {
            input: graph.get_ref(bsn, GraphNodeIdTy::Values),
            output: output.clone(),
        });

        func.push(function::AbsPowerError {
            a: graph.get_ref(self.a.idx, GraphNodeIdTy::Values),
            b: graph.get_ref(self.b.idx, GraphNodeIdTy::Values),
            power: self.power,
            output,
        });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let a = graph.get_ref(self.a.idx, GraphNodeIdTy::Values);
        let b = graph.get_ref(self.b.idx, GraphNodeIdTy::Values);
        let output_grad = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

        let mut func = DeviceFunction::default();

        if let Some(grd) = graph.maybe_get_ref(self.a.idx, GraphNodeIdTy::Gradients) {
            func.push(function::MaybeUpdateBatchSize { input: a.clone(), output: grd.clone() });
            func.push(function::AbsPowerErrorBackward {
                a: a.clone(),
                b: b.clone(),
                c: output_grad.clone(),
                output: grd,
                power: self.power,
            });
        }

        if let Some(grd) = graph.maybe_get_ref(self.b.idx, GraphNodeIdTy::Gradients) {
            func.push(function::MaybeUpdateBatchSize { input: b.clone(), output: grd.clone() });
            func.push(function::AbsPowerErrorBackward {
                a: b.clone(),
                b: a.clone(),
                c: output_grad.clone(),
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
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);
        let a = graph.get_ref(self.a.idx, GraphNodeIdTy::Values);
        let b = graph.get_ref(self.b.idx, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize { input: a.clone(), output: output.clone() });

        func.push(function::CopyOrAddStrided {
            input: a.clone(),
            output: output.clone(),
            input_offset: 0,
            output_offset: 0,
            add: false,
            len_is_out: false,
        });

        func.push(function::CopyOrAddStrided {
            input: b,
            output,
            input_offset: 0,
            output_offset: a.borrow().shape().size(),
            add: false,
            len_is_out: false,
        });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

        let mut func = DeviceFunction::default();

        if let Some(output) = graph.maybe_get_ref(self.a.idx, GraphNodeIdTy::Gradients) {
            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });

            func.push(function::CopyOrAddStrided {
                input: input.clone(),
                output,
                input_offset: 0,
                output_offset: 0,
                add: true,
                len_is_out: true,
            });
        }

        if let Some(output) = graph.maybe_get_ref(self.b.idx, GraphNodeIdTy::Gradients) {
            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });

            let input_offset = graph.get_ref(self.a.idx, GraphNodeIdTy::Values).borrow().shape().size();
            func.push(function::CopyOrAddStrided {
                input,
                output,
                input_offset,
                output_offset: 0,
                add: true,
                len_is_out: true,
            });
        }

        func
    }
}

#[derive(Clone, Debug)]
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
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);
        let a = graph.get_ref(self.a.idx, GraphNodeIdTy::Values);
        let b = graph.get_ref(self.b.idx, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize { input: a.clone(), output: output.clone() });

        func.push(function::PairwiseMul { offset: 0, input: a.clone(), output: output.clone() });

        func.push(function::PairwiseMul { offset: a.borrow().shape().size() / 2, input: b, output });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(output_node, GraphNodeIdTy::Gradients);
        let a = graph.get_ref(self.a.idx, GraphNodeIdTy::Values);
        let offset = a.borrow().shape().size() / 2;

        let mut func = DeviceFunction::default();

        if let Some(output) = graph.maybe_get_ref(self.a.idx, GraphNodeIdTy::Gradients) {
            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });

            func.push(function::PairwiseMulBackward { offset: 0, input: input.clone(), values: a, output });
        }

        if let Some(output) = graph.maybe_get_ref(self.b.idx, GraphNodeIdTy::Gradients) {
            let values = graph.get_ref(self.b.idx, GraphNodeIdTy::Values);

            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });

            func.push(function::PairwiseMulBackward { offset, input, values, output });
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

impl<B: BackendMarker> GraphIROperationCompilable<B> for Select
where
    B::Backend: CoreDeviceOps,
{
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);
        let buckets = graph.get_ref(self.buckets.idx, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize { input: buckets.clone(), output: output.clone() });
        func.push(function::Select { input, output, buckets });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(output) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(output_node, GraphNodeIdTy::Gradients);
            let buckets = graph.get_ref(self.buckets.idx, GraphNodeIdTy::Values);
            let values = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);

            func.push(function::MaybeUpdateBatchSize { input: values, output: output.clone() });
            func.push(function::SelectBackprop { input, output, buckets });
        }

        func
    }
}

#[derive(Clone, Debug)]
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

    fn ancillary_buffers(&self, ir: &GraphIR<B>) -> Result<Vec<(Shape, Option<NonZeroUsize>, bool)>, GraphIRError> {
        let batched = ir.get(self.logits.idx)?.ty().batched;
        Ok(vec![(self.logits.shape, None, batched)])
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for SoftmaxCrossEntropy
where
    B::Backend: CoreDeviceOps,
{
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let logits = graph.get_ref(self.logits.idx, GraphNodeIdTy::Values);
        let targets = graph.get_ref(self.targets.idx, GraphNodeIdTy::Values);
        let smax = graph.get_ref(output_node, GraphNodeIdTy::Ancillary(0));
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize { input: logits.clone(), output: smax.clone() });
        func.push(function::MaybeUpdateBatchSize { input: logits.clone(), output: output.clone() });

        func.push(function::Softmax { input: logits, output: smax.clone() });
        func.push(function::CrossEntropy { a: smax, b: targets, output });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(output) = graph.maybe_get_ref(self.logits.idx, GraphNodeIdTy::Gradients) {
            let softmax = graph.get_ref(output_node, GraphNodeIdTy::Ancillary(0));
            let output_grads = graph.get_ref(output_node, GraphNodeIdTy::Gradients);
            let targets = graph.get_ref(self.targets.idx, GraphNodeIdTy::Values);

            func.push(function::MaybeUpdateBatchSize { input: softmax.clone(), output: output.clone() });
            func.push(function::SoftmaxCrossEntropyBackward { softmax, output_grads, targets, output });
        }

        func
    }
}
