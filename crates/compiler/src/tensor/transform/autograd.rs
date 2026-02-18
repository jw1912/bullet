use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    ir::{NodeId, OpId},
    tensor::{
        IRBuilder, IRNode, IRTrace, TensorIR,
        operation::{CABinary, SubGraph, autograd::AutogradOp},
        transform::{IRTransform, modify::AddOperation},
    },
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowerForward;

impl IRTransform for LowerForward {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        for op in ir.operations() {
            if let Some(AutogradOp { forward, .. }) = op.data().downcast().cloned() {
                ir.replace_op(op.id(), AddOperation::new(op.inputs().to_vec(), Ok(Rc::new(forward))))?;
            }
        }

        Ok(())
    }
}

type GradientMap = HashMap<NodeId, Option<NodeId>>;

#[derive(Debug, Default)]
pub struct TakeGradient {
    root: OpId,
    output_grads: Vec<NodeId>,
    grads: Rc<RefCell<GradientMap>>,
}

impl TakeGradient {
    pub fn new(root: OpId, output_grads: impl Into<Vec<NodeId>>) -> (Self, Rc<RefCell<GradientMap>>) {
        let grads = Rc::<RefCell<GradientMap>>::default();
        (Self { root, output_grads: output_grads.into(), grads: grads.clone() }, grads)
    }
}

impl IRTransform for TakeGradient {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
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
            if let Some(AutogradOp { op, .. }) = operation.data().downcast() {
                let op_ograds = operation.outputs().iter().map(|i| grads.get(i).unwrap().unwrap()).collect::<Vec<_>>();

                // create backwards subgraph
                let builder = IRBuilder::default();
                let inputs = op.inputs().iter().map(|i| builder.add_input(i.size(), i.dtype())).collect::<Vec<_>>();
                let ograds = op_ograds
                    .iter()
                    .map(|&i| {
                        let ty = ir.get_node(i).unwrap().ty();
                        builder.add_input(ty.size(), ty.dtype())
                    })
                    .collect::<Vec<_>>();

                let igrads = op.backward(inputs.clone(), ograds.clone())?;

                // handle not all inputs having gradient and
                // multiple inputs having the same gradient
                let mut igrad_map: HashMap<_, _> = operation.inputs().iter().map(|&inp| (inp, Vec::new())).collect();
                let mut unique_igrads = Vec::new();
                let mut present_igrads = HashSet::new();
                for (&inp, igrad) in operation.inputs().iter().zip(igrads.iter()) {
                    if let Some(ig) = igrad {
                        igrad_map.get_mut(&inp).unwrap().push(ig.node());

                        if present_igrads.insert(ig.node()) {
                            unique_igrads.push(*ig);
                        }
                    }
                }

                let backward = builder.build(&unique_igrads);

                // add backwards subgraph to TensorIR
                let subgraph_inputs = [inputs, ograds].concat().iter().map(IRNode::node).collect();
                let subgraph_outputs: Vec<_> = unique_igrads.iter().map(IRNode::node).collect();
                let subgraph = SubGraph::new(backward, subgraph_inputs, subgraph_outputs.clone())?;
                let new_grads = ir.add_op([operation.inputs(), &op_ograds].concat(), Ok::<_, IRTrace>(subgraph))?;

                // accumulate gradients in actual graph
                let subgraph_map: HashMap<_, _> = subgraph_outputs.iter().zip(new_grads).collect();
                for (input, new_grad_subgraph) in igrad_map {
                    let old_grad = grads.get_mut(&input).unwrap();

                    for subgraph_index in new_grad_subgraph {
                        let new_grad = *subgraph_map.get(&subgraph_index).unwrap();

                        match old_grad {
                            Some(old_grad) => *old_grad = ir.add_binary(*old_grad, new_grad, CABinary::Add)?,
                            None => *old_grad = Some(new_grad),
                        }
                    }
                }
            }
        }

        *self.grads.borrow_mut() = grads;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{
        DType, TType, TValue, TensorOp,
        operation::{CABinaryOp, CopyOp, Input},
        transform::inline::InlineSubgraphs,
    };

    use super::*;

    #[test]
    fn test_axby() -> Result<(), IRTrace> {
        let mut ir = TensorIR::default();

        let ttype = TType::new(1, DType::F32);

        let a = ir.add_input(ttype);
        let b = ir.add_input(ttype);
        let x = ir.add_input(ttype);

        let z = ir.add_op([a, x], AutogradOp::new(CABinaryOp::new(ttype, CABinary::Mul)))?[0];
        let y = ir.add_op([z, b], AutogradOp::new(CABinaryOp::new(ttype, CABinary::Add)))?[0];

        let grad = ir.add_const(TValue::F32(vec![1.0]));

        let (transform, grads) = TakeGradient::new(ir.get_parent_op(y)?, [grad]);
        ir.transform(transform)?;
        let dydx = grads.borrow().get(&x).unwrap().unwrap();
        ir.register_output(dydx);

        ir.transform(LowerForward)?;
        ir.transform(InlineSubgraphs)?;
        ir.optimise()?;

        let ops = ir.ordered_operations()?;
        let mut optys = ops.iter().map(|x| &x.data().0);
        assert_eq!(ops.len(), 2);
        assert!(TensorOp::downcast_rc::<Input>(optys.next().unwrap()).is_some());
        assert!(TensorOp::downcast_rc::<CopyOp>(optys.next().unwrap()).is_some());

        Ok(())
    }

    #[test]
    fn test_pow() -> Result<(), IRTrace> {
        let mut ir = TensorIR::default();

        let ttype = TType::new(1, DType::F32);

        let x = ir.add_input(ttype);

        let y = ir.add_op([x, x], AutogradOp::new(CABinaryOp::new(ttype, CABinary::Mul)))?[0];

        let grad = ir.add_const(TValue::F32(vec![1.0]));

        let (transform, grads) = TakeGradient::new(ir.get_parent_op(y)?, [grad]);
        ir.transform(transform)?;
        ir.transform(LowerForward)?;
        ir.transform(InlineSubgraphs)?;

        let dydx = grads.borrow().get(&x).unwrap().unwrap();
        ir.register_output(dydx);

        ir.optimise()?;

        let ops = ir.ordered_operations()?;
        let mut optys = ops.iter().map(|x| &x.data().0);
        assert_eq!(ops.len(), 2);
        assert!(TensorOp::downcast_rc::<Input>(optys.next().unwrap()).is_some());
        assert!(TensorOp::downcast_rc::<CABinaryOp>(optys.next().unwrap()).is_some());

        Ok(())
    }
}
