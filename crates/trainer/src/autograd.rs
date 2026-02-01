mod core_ops;

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt,
    rc::Rc,
};

use bullet_compiler::{
    IR, IRTrace,
    graph::{NodeId, Op, OpId, OpType, TType, TValue},
    operation::{CABinary, SubGraph},
    prelude::{IRBuilder, IRNode},
    transform::{IRTransform, modify::AddOperation},
};

pub trait Autograd: std::any::Any + fmt::Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<TType>;

    fn forward<'a>(&self, inputs: Vec<IRNode<'a>>) -> Result<Vec<IRNode<'a>>, IRTrace>;

    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace>;

    fn equals(&self, other: &Rc<dyn Autograd>) -> bool;
}

pub trait AutogradOnCoreOp: Clone + OpType + PartialEq {
    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace>;
}

impl<T: AutogradOnCoreOp> Autograd for T {
    fn opname(&self) -> String {
        <T as OpType>::opname(self)
    }

    fn inputs(&self) -> Vec<TType> {
        <T as OpType>::inputs(self)
    }

    fn forward<'a>(&self, inputs: Vec<IRNode<'a>>) -> Result<Vec<IRNode<'a>>, IRTrace> {
        inputs[0].builder().add_op(inputs, self.clone())
    }

    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
        <T as AutogradOnCoreOp>::backward(self, inputs, output_grads)
    }

    fn equals(&self, other: &Rc<dyn Autograd>) -> bool {
        if let Some(other) = AutogradOp::downcast_rc(other) { self == other } else { false }
    }
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

    pub fn new(op: impl Autograd + 'static) -> Result<Self, IRTrace> {
        let op_inputs = op.inputs();

        let builder = IRBuilder::default();
        let inputs = op_inputs.iter().map(|i| builder.add_input(i.size(), i.dtype())).collect::<Vec<_>>();
        let outputs = op.forward(inputs.clone())?;
        let forward = builder.build(&outputs).graph();
        let inputs = inputs.iter().map(IRNode::node).collect();
        let outputs = outputs.iter().map(IRNode::node).collect();
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

                // handle not all inputs having gradient and multiple
                // multiple inputs having the same gradient
                let mut igrad_map = HashMap::new();
                let mut unique_igrads = Vec::new();
                let mut present_igrads = HashSet::new();
                for (&inp, igrad) in operation.inputs().iter().zip(igrads.iter()) {
                    igrad_map.insert(inp, igrad.map(|ig| ig.node()));
                    if let Some(ig) = igrad
                        && present_igrads.insert(ig.node())
                    {
                        unique_igrads.push(*ig);
                    }
                }

                let backward = builder.build(&unique_igrads).graph();

                // add backwards subgraph to IR
                let subgraph_inputs = [inputs, ograds].concat().iter().map(IRNode::node).collect();
                let subgraph_outputs: Vec<_> = unique_igrads.iter().map(IRNode::node).collect();
                let subgraph = SubGraph::new(backward, subgraph_inputs, subgraph_outputs.clone())?;
                let new_grads = ir.add_op([operation.inputs(), &op_ograds].concat(), Ok::<_, IRTrace>(subgraph))?;

                // accumulate gradients in actual graph
                let subgraph_map: HashMap<_, _> = subgraph_outputs.iter().zip(new_grads).collect();
                for (input, new_grad_subgraph) in igrad_map {
                    if let Some(subgraph_index) = new_grad_subgraph {
                        let old_grad = grads.get_mut(&input).unwrap();
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
    use bullet_compiler::{
        graph::{DType, Input},
        operation::{CABinaryOp, CopyOp},
        transform::inline::InlineSubgraphs,
    };

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

        let (transform, grads) = TakeGradient::new(ir.get_parent_op(y)?, [grad]);
        ir.transform(transform)?;
        ir.transform(LowerForward)?;
        ir.transform(InlineSubgraphs)?;

        let dydx = grads.borrow().get(&x).unwrap().unwrap();
        ir.register_output(dydx);

        ir.optimise()?;

        let ops = ir.ordered_operations()?;
        let mut optys = ops.iter().map(Op::op);
        assert!(Op::downcast_rc::<Input>(optys.next().unwrap()).is_some());
        assert!(Op::downcast_rc::<CopyOp>(optys.next().unwrap()).is_some());

        Ok(())
    }
}
