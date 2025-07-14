use crate::graph::{
    instruction,
    ir::{
        node::AnnotatedNode,
        operation::{unary::Reduce, util, GraphIROperation, GraphIROperationCompilable, GraphIROperationError},
        shape::Shape,
        BackendMarker, GraphIR, GraphIRError, GraphIRNodeInfo,
    },
    GraphFunction, NodeId, NodeIdTy,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearCombination {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
    pub alpha: f32,
    pub beta: f32,
}

impl<B: BackendMarker> GraphIROperation<B> for LinearCombination {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.a, self.b]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.a, true)?;
        util::check_dense_eq(ir, &self.b, true)?;

        if self.a.shape == self.b.shape {
            Ok(self.a.shape)
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![self.a.shape, self.b.shape])))
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for LinearCombination {
    fn forward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let output = NodeId::new(output_node, NodeIdTy::Values);
        let bsn = util::batch_size_node(node_info, &[self.a, self.b]);

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input: NodeId::new(bsn, NodeIdTy::Values), output });

        let mut push = |input_mul, output_mul, node| {
            if !node_info.get(output_node).unwrap().batched || node_info.get(node).unwrap().batched {
                func.push(instruction::LinearCombination {
                    input_mul,
                    output_mul,
                    input: NodeId::new(node, NodeIdTy::Values),
                    output,
                });
            } else {
                func.push(instruction::LinearCombinationSplat {
                    input_mul,
                    output_mul,
                    input: NodeId::new(node, NodeIdTy::Values),
                    output,
                });
            }
        };

        push(self.alpha, 0.0, self.a.idx);
        push(self.beta, 1.0, self.b.idx);

        func
    }

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let input = NodeId::new(output_node, NodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        let mut push = |input_mul, output_mul, node| {
            if node_info.get(node).unwrap().requires_grad {
                let output = NodeId::new(node, NodeIdTy::Gradients);

                func.push(instruction::MaybeUpdateBatchSize { input: NodeId::new(node, NodeIdTy::Values), output });

                if !node_info.get(output_node).unwrap().batched || node_info.get(node).unwrap().batched {
                    func.push(instruction::LinearCombination { input_mul, output_mul, input, output });
                } else {
                    func.push(instruction::ReduceAcrossBatch {
                        input_mul,
                        output_mul,
                        input,
                        output,
                        reduction: Reduce::Sum,
                    });
                }
            }
        };

        push(self.alpha, 1.0, self.a.idx);
        push(self.beta, 1.0, self.b.idx);

        func
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbsPowerError {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
    pub power: f32,
}

impl<B: BackendMarker> GraphIROperation<B> for AbsPowerError {
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
    fn forward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let output = NodeId::new(output_node, NodeIdTy::Values);
        let bsn = util::batch_size_node(node_info, &[self.a, self.b]);

        let mut func = GraphFunction::default();

        func.push(instruction::MaybeUpdateBatchSize { input: NodeId::new(bsn, NodeIdTy::Values), output });

        func.push(instruction::AbsPowerError {
            a: NodeId::new(self.a.idx, NodeIdTy::Values),
            b: NodeId::new(self.b.idx, NodeIdTy::Values),
            power: self.power,
            output,
        });

        func
    }

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let a = NodeId::new(self.a.idx, NodeIdTy::Values);
        let b = NodeId::new(self.b.idx, NodeIdTy::Values);
        let output_grad = NodeId::new(output_node, NodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        if node_info.get(self.a.idx).unwrap().requires_grad {
            let grd = NodeId::new(self.a.idx, NodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input: a, output: grd });
            func.push(instruction::AbsPowerErrorBackward { a, b, c: output_grad, output: grd, power: self.power });
        }

        if node_info.get(self.b.idx).unwrap().requires_grad {
            let grd = NodeId::new(self.b.idx, NodeIdTy::Gradients);

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

#[derive(Debug)]
pub struct Concat {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperation<B> for Concat {
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
    fn forward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let output = NodeId::new(output_node, NodeIdTy::Values);
        let a = NodeId::new(self.a.idx, NodeIdTy::Values);
        let b = NodeId::new(self.b.idx, NodeIdTy::Values);

        let mut func = GraphFunction::default();

        let ainfo = node_info.get(self.a.idx).unwrap();

        assert_eq!(ainfo.batched, node_info.get(self.b.idx).unwrap().batched);

        func.push(instruction::MaybeUpdateBatchSize { input: a, output });

        func.push(instruction::CopyOrAddStrided {
            input: a,
            output,
            input_offset: 0,
            output_offset: 0,
            add: false,
            backprop: false,
        });

        func.push(instruction::CopyOrAddStrided {
            input: b,
            output,
            input_offset: 0,
            output_offset: ainfo.shape.size(),
            add: false,
            backprop: false,
        });

        func
    }

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<B::Backend> {
        let input = NodeId::new(output_node, NodeIdTy::Gradients);

        let ainfo = node_info.get(self.a.idx).unwrap();
        let binfo = node_info.get(self.b.idx).unwrap();

        let mut func = GraphFunction::default();

        if ainfo.requires_grad {
            let output = NodeId::new(self.a.idx, NodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output });

            func.push(instruction::CopyOrAddStrided {
                input,
                output,
                input_offset: 0,
                output_offset: 0,
                add: true,
                backprop: true,
            });
        }

        if binfo.requires_grad {
            let output = NodeId::new(self.b.idx, NodeIdTy::Gradients);

            func.push(instruction::MaybeUpdateBatchSize { input, output });

            func.push(instruction::CopyOrAddStrided {
                input,
                output,
                input_offset: ainfo.shape.size(),
                output_offset: 0,
                add: true,
                backprop: true,
            });
        }

        func
    }
}

#[derive(Debug)]
pub struct Select {
    pub input: AnnotatedNode,
    pub buckets: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperation<B> for Select {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input, self.buckets]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        util::check_dense_eq(ir, &self.buckets, false)?;
        util::check_same_batching(ir, &[&self.input, &self.buckets])?;
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
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, _output_node: usize) -> GraphFunction<B::Backend> {
        todo!()
    }

    fn backward_pass(&self, _node_info: &GraphIRNodeInfo, _output_node: usize) -> GraphFunction<B::Backend> {
        todo!()
    }
}
