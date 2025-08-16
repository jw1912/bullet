use acyclib::graph::NodeId;

use crate::graph::{
    instruction,
    ir::{
        node::AnnotatedNode,
        operation::{util, GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError},
        shape::Shape,
        BackendMarker, GraphIR, GraphIRError,
    },
    GraphFunction, GraphNodeId, GraphNodeIdTy,
};

/// List of supported activation functions.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffableFromOutput {
    Identity = 0,
    ReLU = 1,
    CReLU = 2,
    SCReLU = 3,
    SqrReLU = 4,
    Sigmoid = 5,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unary {
    pub input: AnnotatedNode,
    pub op: UnaryOp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnaryOp {
    DiffableFromOutput(DiffableFromOutput),
    Add(f32),
    Mul(f32),
    AbsPow(f32),
}

impl<B: BackendMarker> GraphIROperationBase<B> for Unary {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        Ok(self.input.shape)
    }

    fn shorthand(&self) -> String {
        match self.op {
            UnaryOp::AbsPow(p) => format!("|x|^{p}"),
            UnaryOp::Add(x) => format!("+ {x}"),
            UnaryOp::Mul(x) => format!("* {x}"),
            UnaryOp::DiffableFromOutput(act) => format!("{act:?}"),
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Unary {
    fn forward_pass(&self, _ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let input = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Values);
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);

        let mut func = GraphFunction::default();
        func.push(instruction::MaybeUpdateBatchSize { input, output });
        func.push(instruction::Unary { input, output, op: self.op });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        assert!(ir.get(output_node).unwrap().ty().requires_grad);

        let mut func = GraphFunction::default();

        if ir.get(self.input.idx).unwrap().ty().requires_grad {
            let input = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Values);
            let input_grad = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output: input_grad });

            func.push(instruction::UnaryBackward {
                input,
                input_grad,
                output_grad: GraphNodeId::new(output_node, GraphNodeIdTy::Gradients),
                op: self.op,
            });
        }

        func
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reduce {
    Sum,
    Avg,
}

#[derive(Debug)]
pub struct ReduceAcrossBatch {
    pub input: AnnotatedNode,
    pub reduction: Reduce,
}

impl<B: BackendMarker> GraphIROperationBase<B> for ReduceAcrossBatch {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_batched(&self, _: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(false)
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        if util::check_not_batched(ir, &self.input).is_ok() {
            return Err(GraphIRError::Op(GraphIROperationError::MismatchedBatching));
        }

        Ok(self.input.shape)
    }

    fn shorthand(&self) -> String {
        format!("Reduce{:?}AcrossBatch", self.reduction)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for ReduceAcrossBatch {
    fn forward_pass(&self, _ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let input = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Values);
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);

        let mut func = GraphFunction::default();

        func.push(instruction::ReduceAcrossBatch {
            input,
            output,
            input_mul: 1.0,
            output_mul: 0.0,
            reduction: self.reduction,
        });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let info = ir.get(output_node).unwrap().ty();
        assert!(info.requires_grad);
        assert!(!info.batched);

        let input = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Values);
        let input_grad = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Gradients);
        let output_grad = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input, output: input_grad });

        if ir.get(self.input.idx).unwrap().ty().requires_grad {
            func.push(instruction::SplatAcrossBatch {
                input: output_grad,
                output: input_grad,
                reduction: self.reduction,
                input_mul: 1.0,
                output_mul: 1.0,
            });
        }

        func
    }
}

#[derive(Debug)]
pub struct PairwiseMul {
    pub input: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperationBase<B> for PairwiseMul {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        let is = self.input.shape;

        if is.rows() % 2 == 0 {
            Ok(Shape::new(is.rows() / 2, is.cols()))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(is)))
        }
    }

    fn shorthand(&self) -> String {
        "PairwiseMul".to_string()
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for PairwiseMul {
    fn forward_pass(&self, _ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let input = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Values);
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input, output });
        func.push(instruction::PairwiseMul { offset: 0, input, output });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let mut func = GraphFunction::default();

        if ir.get(self.input.idx).unwrap().ty().requires_grad {
            let input = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);
            let output = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output });
            func.push(instruction::PairwiseMulBackward {
                offset: 0,
                values: GraphNodeId::new(self.input.idx, GraphNodeIdTy::Values),
                input,
                output,
            });
        }

        func
    }
}

#[derive(Debug)]
pub struct Slice {
    pub input: AnnotatedNode,
    pub start: usize,
    pub end: usize,
}

impl<B: BackendMarker> GraphIROperationBase<B> for Slice {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        let is = self.input.shape;
        if self.end > self.start && self.end <= is.rows() && is.cols() == 1 {
            Ok(Shape::new(self.end - self.start, 1))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::OutOfBounds(is, [self.start, self.end])))
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Slice {
    fn forward_pass(&self, _ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let input = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Values);
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);

        let mut func = GraphFunction::default();
        func.push(instruction::MaybeUpdateBatchSize { input, output });
        func.push(instruction::CopyOrAddStrided {
            input,
            output,
            input_offset: self.start,
            output_offset: 0,
            add: false,
            len_is_out: true,
        });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let mut func = GraphFunction::default();

        if ir.get(self.input.idx).unwrap().ty().requires_grad {
            let input = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);
            let output = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output });
            func.push(instruction::CopyOrAddStrided {
                input,
                output,
                input_offset: 0,
                output_offset: self.start,
                add: true,
                len_is_out: false,
            });
        }

        func
    }
}

#[derive(Debug)]
pub struct ToDense(pub AnnotatedNode);

impl<B: BackendMarker> GraphIROperationBase<B> for ToDense {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.0]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.0, false)?;
        Ok(self.0.shape)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for ToDense {
    fn forward_pass(&self, _ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let input = GraphNodeId::new(self.0.idx, GraphNodeIdTy::Values);
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);

        let mut func = GraphFunction::default();
        func.push(instruction::MaybeUpdateBatchSize { input, output });
        func.push(instruction::SparseToDense { input, output });

        func
    }

    fn backward_pass(&self, _ir: &GraphIR<B>, _output_node: NodeId) -> GraphFunction<B::Backend> {
        GraphFunction::default()
    }
}

#[derive(Debug)]
pub struct Copy {
    pub input: AnnotatedNode,
    pub stop_grad: bool,
}

impl<B: BackendMarker> GraphIROperationBase<B> for Copy {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        Ok(self.input.shape)
    }

    fn output_requires_grad(&self, ir: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(!self.stop_grad && ir.get(self.input.idx).unwrap().ty().requires_grad)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Copy {
    fn forward_pass(&self, _ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let input = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Values);
        let output = GraphNodeId::new(output_node, GraphNodeIdTy::Values);

        let mut func = GraphFunction::default();
        func.push(instruction::MaybeUpdateBatchSize { input, output });
        func.push(instruction::LinearCombination { input_mul: 1.0, output_mul: 0.0, input, output });

        func
    }

    fn backward_pass(&self, ir: &GraphIR<B>, output_node: NodeId) -> GraphFunction<B::Backend> {
        let mut func = GraphFunction::default();

        if !self.stop_grad && ir.get(self.input.idx).unwrap().ty().requires_grad {
            let input = GraphNodeId::new(output_node, GraphNodeIdTy::Gradients);
            let output = GraphNodeId::new(self.input.idx, GraphNodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output });
            func.push(instruction::LinearCombination { input_mul: 1.0, output_mul: 1.0, input, output });
        }

        func
    }
}
