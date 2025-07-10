use crate::{
    backend::device::Device,
    graph::{
        instruction,
        ir::{
            node::AnnotatedNode,
            operation::{util, GraphIROperation, GraphIROperationCompile, GraphIROperationError},
            shape::Shape,
            GraphIR, GraphIRError, GraphIRNodeInfo,
        },
        GraphFunction, NodeId, NodeIdTy,
    },
};

#[derive(Debug)]
pub struct Matmul {
    pub a: AnnotatedNode,
    pub b: AnnotatedNode,
    pub transa: bool,
    pub transb: bool,
}

impl GraphIROperation for Matmul {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.a, self.b]
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.a, true)?;
        util::check_dense_eq(ir, &self.b, true)?;
        util::check_matmul(self.a.shape.maybe_transpose(self.transa), self.b.shape.maybe_transpose(self.transb))
            .map_err(GraphIRError::Op)
    }
}

impl<D: Device> GraphIROperationCompile<D> for Matmul {
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<D> {
        let mut func = GraphFunction::default();

        func.push(instruction::Matmul {
            alpha: 1.0,
            beta: 0.0,
            input_a: NodeId::new(self.a.idx, NodeIdTy::Values),
            input_b: NodeId::new(self.b.idx, NodeIdTy::Values),
            trans_a: self.transa,
            trans_b: self.transb,
            shape_a: self.a.shape,
            shape_b: self.b.shape,
            output: NodeId::new(output_node, NodeIdTy::Values),
        });

        func
    }

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<D> {
        let mut func = GraphFunction::default();

        let shape_o = self.a.shape.maybe_transpose(self.transa) * self.b.shape.maybe_transpose(self.transb);

        if node_info.get(self.a.idx).unwrap().requires_grad {
            let instr = if self.transa {
                instruction::Matmul {
                    alpha: 1.0,
                    beta: 1.0,
                    output: NodeId::new(self.a.idx, NodeIdTy::Gradients),
                    input_a: NodeId::new(self.b.idx, NodeIdTy::Values),
                    shape_a: self.b.shape,
                    trans_a: self.transb,
                    input_b: NodeId::new(output_node, NodeIdTy::Gradients),
                    shape_b: shape_o,
                    trans_b: true,
                }
            } else {
                instruction::Matmul {
                    alpha: 1.0,
                    beta: 1.0,
                    output: NodeId::new(self.a.idx, NodeIdTy::Gradients),
                    input_a: NodeId::new(output_node, NodeIdTy::Gradients),
                    shape_a: shape_o,
                    trans_a: false,
                    input_b: NodeId::new(self.b.idx, NodeIdTy::Values),
                    shape_b: self.b.shape,
                    trans_b: !self.transb,
                }
            };

            func.push(instr);
        }

        if node_info.get(self.b.idx).unwrap().requires_grad {
            let instr = if self.transb {
                instruction::Matmul {
                    alpha: 1.0,
                    beta: 1.0,
                    output: NodeId::new(self.b.idx, NodeIdTy::Gradients),
                    input_a: NodeId::new(output_node, NodeIdTy::Gradients),
                    shape_a: shape_o,
                    trans_a: true,
                    input_b: NodeId::new(self.a.idx, NodeIdTy::Values),
                    shape_b: self.a.shape,
                    trans_b: self.transa,
                }
            } else {
                instruction::Matmul {
                    alpha: 1.0,
                    beta: 1.0,
                    output: NodeId::new(self.b.idx, NodeIdTy::Gradients),
                    input_a: NodeId::new(self.a.idx, NodeIdTy::Values),
                    shape_a: self.a.shape,
                    trans_a: !self.transa,
                    input_b: NodeId::new(output_node, NodeIdTy::Gradients),
                    shape_b: shape_o,
                    trans_b: false,
                }
            };

            func.push(instr);
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

impl GraphIROperation for Affine {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.weights, self.biases, self.inputs]
    }

    fn output_shape(&self, ir: &GraphIR) -> Result<Shape, GraphIRError> {
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

impl<D: Device> GraphIROperationCompile<D> for Affine {
    fn forward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<D> {
        let matmul = Matmul { a: self.weights, b: self.inputs, transa: false, transb: false };

        let mut func = matmul.forward_pass(node_info, output_node);

        todo!();

        func
    }

    fn backward_pass(&self, node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<D> {
        let matmul = Matmul { a: self.weights, b: self.inputs, transa: false, transb: false };

        let mut func = matmul.backward_pass(node_info, output_node);

        if node_info.get(self.biases.idx).unwrap().requires_grad {
            todo!()
        }

        func
    }
}
