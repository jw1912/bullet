use crate::graph::{
    instruction,
    ir::{
        node::AnnotatedNode,
        operation::{util, GraphIROperation, GraphIROperationCompilable, GraphIROperationError},
        shape::Shape,
        BackendMarker, GraphIR, GraphIRError, GraphIRNodeInfo,
    },
    GraphFunction, NodeId, NodeIdTy,
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

impl<B: BackendMarker> GraphIROperation<B> for Unary {
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
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

        let mut func = GraphFunction::default();
        func.push(instruction::MaybeUpdateBatchSize { input, output });
        func.push(instruction::Unary { input, output, op: self.op });

        func
    }

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        assert!(node_info.get(output_node).unwrap().requires_grad);

        let mut func = GraphFunction::default();

        if node_info.get(self.input.idx).unwrap().requires_grad {
            let input = NodeId::new(self.input.idx, NodeIdTy::Values);
            let input_grad = NodeId::new(self.input.idx, NodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output: input_grad });

            func.push(instruction::UnaryBackward {
                input,
                input_grad,
                output_grad: NodeId::new(output_node, NodeIdTy::Gradients),
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

impl<B: BackendMarker> GraphIROperation<B> for ReduceAcrossBatch {
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
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

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

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let info = node_info.get(output_node).unwrap();
        assert!(info.requires_grad);
        assert!(!info.batched);

        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let input_grad = NodeId::new(self.input.idx, NodeIdTy::Gradients);
        let output_grad = NodeId::new(output_node, NodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input, output: input_grad });

        if node_info.get(self.input.idx).unwrap().requires_grad {
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
    pub post_concat: bool,
}

impl<B: BackendMarker> GraphIROperation<B> for PairwiseMul {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        let is = self.input.shape;
        let min = 2 + 2 * usize::from(self.post_concat);

        if is.rows() % min == 0 {
            Ok(Shape::new(is.rows() / 2, is.cols()))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(is)))
        }
    }

    fn shorthand(&self) -> String {
        if self.post_concat {
            "PairwiseMulPostConcat".to_string()
        } else {
            "PairwiseMul".to_string()
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for PairwiseMul {
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

        let mut func = GraphFunction::default();
        func.push(instruction::MaybeUpdateBatchSize { input, output });
        func.push(instruction::PairwiseMul { post_concat: self.post_concat, input, output });

        func
    }

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let mut func = GraphFunction::default();

        if node_info.get(self.input.idx).unwrap().requires_grad {
            let input = NodeId::new(output_node, NodeIdTy::Gradients);
            let output = NodeId::new(self.input.idx, NodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output });
            func.push(instruction::PairwiseMulBackward {
                post_concat: self.post_concat,
                values: NodeId::new(self.input.idx, NodeIdTy::Values),
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

impl<B: BackendMarker> GraphIROperation<B> for Slice {
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
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

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

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let mut func = GraphFunction::default();

        if node_info.get(self.input.idx).unwrap().requires_grad {
            let input = NodeId::new(output_node, NodeIdTy::Gradients);
            let output = NodeId::new(self.input.idx, NodeIdTy::Gradients);

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

impl<B: BackendMarker> GraphIROperation<B> for ToDense {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.0]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.0, false)?;
        Ok(self.0.shape)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for ToDense {
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let input = NodeId::new(self.0.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

        let mut func = GraphFunction::default();
        func.push(instruction::MaybeUpdateBatchSize { input, output });
        func.push(instruction::SparseToDense { input, output });

        func
    }

    fn backward_pass(&self, _node_info: &GraphIRNodeInfo, _output_node: usize) -> GraphFunction<B::Backend> {
        GraphFunction::default()
    }
}

#[derive(Debug)]
pub struct Copy {
    pub input: AnnotatedNode,
    pub stop_grad: bool,
}

impl<B: BackendMarker> GraphIROperation<B> for Copy {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        Ok(self.input.shape)
    }

    fn output_requires_grad(&self, ir: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(!self.stop_grad && ir.get(self.input.idx).unwrap().info.requires_grad)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Copy {
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

        let mut func = GraphFunction::default();
        func.push(instruction::MaybeUpdateBatchSize { input, output });
        func.push(instruction::LinearCombination { input_mul: 1.0, output_mul: 0.0, input, output });

        func
    }

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let mut func = GraphFunction::default();

        if !self.stop_grad && node_info.get(self.input.idx).unwrap().requires_grad {
            let input = NodeId::new(output_node, NodeIdTy::Gradients);
            let output = NodeId::new(self.input.idx, NodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output });
            func.push(instruction::LinearCombination { input_mul: 1.0, output_mul: 1.0, input, output });
        }

        func
    }
}
