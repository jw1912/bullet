mod binary;
mod unary;

use std::{collections::HashMap, fmt, rc::Rc};

use bullet_compiler::{
    IR, IRTrace,
    graph::{GraphError, NodeId, Op, OpId, OpType, TType, TValue},
    operation::{CABinary, SubGraph},
    prelude::{ProgramBuilder, ProgramNode},
    transform::{IRTransform, modify::AddOperation},
};

pub trait Autograd: std::any::Any + fmt::Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<TType>;

    fn forward<'a>(&self, inputs: &[ProgramNode<'a>]) -> Vec<ProgramNode<'a>>;

    fn backward<'a>(
        &self,
        inputs: &[ProgramNode<'a>],
        output_grads: &[ProgramNode<'a>],
    ) -> Vec<Option<ProgramNode<'a>>>;

    fn equals(&self, other: &Rc<dyn Autograd>) -> bool;
}

#[derive(Clone, Debug)]
pub struct AutogradOp {
    op: Rc<dyn Autograd>,
    forward: SubGraph,
}

impl AutogradOp {
    pub fn downcast_rc<T: Autograd>(input: &Rc<dyn Autograd>) -> Option<&T> {
        let op: &dyn std::any::Any = input.as_ref();
        op.downcast_ref::<T>()
    }

    pub fn downcast<T: Autograd>(&self) -> Option<&T> {
        Self::downcast_rc::<T>(&self.op)
    }

    pub fn new(op: impl Autograd + 'static) -> Result<Self, GraphError> {
        let op_inputs = op.inputs();

        let builder = ProgramBuilder::default();
        let inputs = op_inputs.iter().map(|i| builder.add_input(i.size(), i.dtype())).collect::<Vec<_>>();
        let outputs = op.forward(&inputs);
        let forward = builder.build(&outputs).graph();
        let inputs = inputs.iter().map(ProgramNode::node).collect();
        let outputs = outputs.iter().map(ProgramNode::node).collect();
        let forward = SubGraph::new(forward, inputs, outputs)?;

        Ok(Self { op: Rc::new(op), forward })
    }
}

impl OpType for AutogradOp {
    fn opname(&self) -> String {
        format!("autograd.{}", self.op.opname())
    }

    fn inputs(&self) -> Vec<TType> {
        self.op.inputs()
    }

    fn outputs(&self) -> Vec<TType> {
        self.forward.outputs()
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(AutogradOp { op, .. }) = Op::downcast_rc(other) { self.op.equals(op) } else { false }
    }

    fn evaluate(&self, inputs: Vec<&TValue>, outputs: Vec<&mut TValue>) {
        self.forward.evaluate(inputs, outputs);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowerForward;

impl IRTransform for LowerForward {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        for op in ir.operations() {
            if let Some(AutogradOp { forward, .. }) = op.downcast().cloned() {
                ir.replace_op(op.id(), AddOperation(op.inputs().to_vec(), Ok(Rc::new(forward))))?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TakeGradient {
    root: OpId,
    output_grads: Vec<NodeId>,
}

impl IRTransform for TakeGradient {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        let ops = ir.get_dependent_ops_set(self.root)?;

        let root_outputs = ir.get_op(self.root)?.outputs().to_vec();
        let root_grads = self.output_grads.clone().into_iter().map(Option::Some);
        let mut grads: HashMap<_, _> = root_outputs.into_iter().zip(root_grads).collect();

        for &op in &ops {
            for &input in ir.get_op(op)?.inputs() {
                grads.insert(input, None);
            }
        }

        for operation in ir.ordered_operations()?.iter().rev().filter(|operation| ops.contains(&operation.id())) {
            if let Some(AutogradOp { op, .. }) = operation.downcast() {
                let op_ograds = operation.outputs().iter().map(|i| grads.get(i).unwrap().unwrap()).collect::<Vec<_>>();

                let builder = ProgramBuilder::default();
                let inputs = op.inputs().iter().map(|i| builder.add_input(i.size(), i.dtype())).collect::<Vec<_>>();
                let ograds = op_ograds
                    .iter()
                    .map(|&i| {
                        let ty = ir.get_node(i).unwrap().ty();
                        builder.add_input(ty.size(), ty.dtype())
                    })
                    .collect::<Vec<_>>();

                let igrads = op.backward(&inputs, &ograds);
                let some_igrads = igrads.iter().cloned().flatten().collect::<Vec<_>>();
                let backward = builder.build(&some_igrads).graph();

                let inputs = [inputs, ograds].concat().iter().map(ProgramNode::node).collect();
                let outputs = some_igrads.iter().map(ProgramNode::node).collect();
                let subgraph = SubGraph::new(backward, inputs, outputs)?;

                let new_grads = ir.add_op([operation.inputs(), &op_ograds].concat(), Ok::<_, IRTrace>(subgraph))?;

                let mut i = 0;
                for (input, new_grad) in operation.inputs().iter().zip(igrads) {
                    if new_grad.is_some() {
                        let new_grad = new_grads[i];
                        i += 1;
                        let old_grad = grads.get_mut(input).unwrap();

                        match old_grad {
                            Some(old_grad) => *old_grad = ir.add_binary(*old_grad, new_grad, CABinary::Add)?,
                            None => *old_grad = Some(new_grad),
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use bullet_compiler::{graph::DType, operation::CABinaryOp, transform::inline::InlineSubgraphs};

    use super::*;

    #[test]
    fn test_axby() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let ttype = TType::new(1, DType::F32);

        let a = ir.add_input(ttype);
        let b = ir.add_input(ttype);
        let x = ir.add_input(ttype);

        let z = ir.add_op([a, x], AutogradOp::new(CABinaryOp::new(ttype, CABinary::Mul)))?[0];
        let y = ir.add_op([z, b], AutogradOp::new(CABinaryOp::new(ttype, CABinary::Add)))?[0];

        let grad = ir.add_const(TValue::F32(vec![1.0]));

        ir.transform(TakeGradient { root: ir.get_parent_op(y)?, output_grads: vec![grad] })?;
        ir.transform(LowerForward)?;
        ir.transform(InlineSubgraphs)?;

        println!("{ir}");

        Ok(())
    }
}
