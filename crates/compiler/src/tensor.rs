pub mod ansi;
pub mod builder;
pub mod operation;
mod pattern;
pub mod transform;
mod ttype;

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt,
    rc::Rc,
};

use crate::ir::{IR, IRError, Node, NodeId, Op, OpId, TypeSystem};

use operation::{
    BroadcastAcrossDimension, CABinary, CABinaryOp, Constant, CopyOp, Input, ScalarConstant, Unary, UnaryOp,
};
use transform::{
    CanonicalisePass, IRTransform,
    eliminate::{EliminateCopies, EliminateUnusedOperations},
    modify::{AddOperation, RemoveOperation, ReplaceInput, ReplaceOperation, SwapOutputs},
};

use ansi::Ansi;
pub use operation::{OpType, TensorOp};
pub use ttype::{DType, DValue, Shape, Size, TType, TValue};

#[derive(Clone, Copy, Debug, Default)]
pub struct Tensor;

impl TypeSystem for Tensor {
    type Type = TType;
    type OpData = TensorOp;
}

#[derive(Clone, Debug, Default)]
pub struct TensorIR {
    ir: IR<Tensor>,
    outputs: HashSet<NodeId>,
}

impl TensorIR {
    pub fn ir(&self) -> &IR<Tensor> {
        &self.ir
    }

    pub fn ir_mut(&mut self) -> &mut IR<Tensor> {
        &mut self.ir
    }

    pub fn evaluate(&self, inputs: impl Into<HashMap<NodeId, TValue>>) -> Result<HashMap<NodeId, TValue>, IRTrace> {
        let mut values: HashMap<_, _> =
            inputs.into().into_iter().map(|(id, tensor)| (id, RefCell::new(tensor))).collect();

        let mut vars = HashSet::new();

        for (id, tensor) in &values {
            let op = self.get_op(self.get_parent_op(*id)?)?;
            if !op.data().is_input() {
                return Err("Seeded non-leaf node!".into());
            }

            let concrete_size = tensor.borrow().size();
            let size = self.get_node(*id)?.ty().size();

            if let Some(var) = size.get_var_size(concrete_size) {
                vars.insert(var);
            }
        }

        let var = match vars.len() {
            0 => 1,
            1 => *vars.iter().next().unwrap(),
            _ => return Err(format!("Mismatching batch sizes in inputs: {vars:?}").into()),
        };

        for op in self.ordered_operations()? {
            for &output in op.outputs() {
                let ty = self.get_node(output)?.ty();
                let size = ty.size().evaluate(var);
                let tensor = TValue::zeros(ty.dtype(), size);
                let is_prev = values.contains_key(&output);

                if !op.data().is_input() {
                    assert!(values.insert(output, RefCell::new(tensor)).is_none(), "Cannot happen!");
                } else if !is_prev {
                    return Err("Input node not seeded!".into());
                }
            }

            let op_inputs = op
                .inputs()
                .iter()
                .map(|i| values.get(i).map(|i| i.borrow()))
                .collect::<Option<Vec<_>>>()
                .ok_or("Input missing!")?;

            let mut op_outputs = op
                .outputs()
                .iter()
                .map(|i| values.get(i).map(|i| i.borrow_mut()))
                .collect::<Option<Vec<_>>>()
                .ok_or("Output missing!")?;

            op.data()
                .0
                .evaluate(op_inputs.iter().map(|x| &**x).collect(), op_outputs.iter_mut().map(|x| &mut **x).collect());
        }

        Ok(values.into_iter().filter_map(|x| self.is_output(x.0).then(|| (x.0, x.1.into_inner()))).collect())
    }

    pub fn transform(&mut self, transform: impl IRTransform) -> Result<(), IRTrace> {
        self.transform_dyn(Rc::new(transform))
    }

    pub fn transform_dyn(&mut self, transform: Rc<dyn IRTransform>) -> Result<(), IRTrace> {
        let ir = Box::new(self.ir().clone());
        transform.apply(self).map_err(|err| IRTrace::Frame(ir, transform, Rc::new(err)))?;
        Ok(())
    }

    pub fn ordered_operations(&self) -> Result<Vec<Op<Tensor>>, IRTrace> {
        let ids = self.ir.topo_order_ops()?;
        ids.into_iter().map(|id| self.ir.op(id).map_err(IRTrace::Root).cloned()).collect()
    }

    pub fn operations(&self) -> Vec<Op<Tensor>> {
        self.ir.operations().cloned().collect()
    }

    pub fn num_nontrivial_operations(&self) -> Result<usize, IRTrace> {
        let mut count = 0;

        for op in self.ordered_operations()? {
            let data = op.data();
            if !data.is_input() && data.downcast::<Constant>().is_none() && data.downcast::<ScalarConstant>().is_none()
            {
                count += 1;
            }
        }

        Ok(count)
    }

    pub fn get_parent_op(&self, node: NodeId) -> Result<OpId, IRTrace> {
        self.ir.parent_op(node).map_err(IRTrace::Root)
    }

    pub fn get_op(&self, op: OpId) -> Result<&Op<Tensor>, IRTrace> {
        self.ir.op(op).map_err(IRTrace::Root)
    }

    pub fn get_op_mut(&mut self, op: OpId) -> Result<&mut Op<Tensor>, IRTrace> {
        self.ir.op_mut(op).map_err(IRTrace::Root)
    }

    pub fn get_node(&self, node: NodeId) -> Result<&Node<Tensor>, IRTrace> {
        self.ir.node(node).map_err(IRTrace::Root)
    }

    pub fn get_node_mut(&mut self, node: NodeId) -> Result<&mut Node<Tensor>, IRTrace> {
        self.ir.node_mut(node).map_err(IRTrace::Root)
    }

    pub fn is_output(&self, node: NodeId) -> bool {
        self.outputs.contains(&node)
    }

    pub fn is_copy(&self, node: NodeId) -> Result<Option<NodeId>, IRTrace> {
        let op = self.get_op(self.get_parent_op(node)?)?;
        Ok(op.data().downcast::<CopyOp>().map(|_| op.inputs()[0]))
    }

    pub fn are_copies(&self, a: NodeId, b: NodeId) -> Result<bool, IRTrace> {
        Ok(self.is_copy(a)? == Some(b) || self.is_copy(b)? == Some(a))
    }

    pub fn parent_op<T: OpType>(&self, node: NodeId) -> Result<Option<&T>, IRTrace> {
        let id = self.get_parent_op(node)?;
        let op = self.get_op(id)?;
        Ok(op.data().downcast::<T>())
    }

    pub fn is_input(&self, node: NodeId) -> Result<bool, IRTrace> {
        self.parent_op::<Input>(node).map(|x| x.is_some())
    }

    pub fn is_constant(&self, node: NodeId) -> Result<bool, IRTrace> {
        Ok(self.parent_op::<Constant>(node)?.is_some() || self.parent_op::<ScalarConstant>(node)?.is_some())
    }

    pub fn check_valid(&self) -> Result<(), IRTrace> {
        self.ir.check_valid().map_err(IRTrace::Root)
    }

    pub fn register_output(&mut self, node: NodeId) {
        self.outputs.insert(node);
    }

    pub fn unregister_output(&mut self, node: NodeId) {
        self.outputs.remove(&node);
    }

    pub fn add_dyn_op(
        &mut self,
        inputs: impl AsRef<[NodeId]>,
        op: Result<Rc<dyn OpType>, IRTrace>,
    ) -> Result<Vec<NodeId>, IRTrace> {
        let inputs = inputs.as_ref().to_vec();
        let transform = AddOperation::new(inputs, op);
        self.transform(transform.clone()).map(|_| transform.outputs())
    }

    pub fn add_op(
        &mut self,
        inputs: impl AsRef<[NodeId]>,
        op: Result<impl OpType, impl Into<IRTrace>>,
    ) -> Result<Vec<NodeId>, IRTrace> {
        fn convert(x: impl OpType) -> Rc<dyn OpType> {
            Rc::new(x)
        }

        let op = op.map(convert).map_err(|e| e.into());

        self.add_dyn_op(inputs, op)
    }

    #[must_use]
    pub fn add_input(&mut self, ty: TType) -> NodeId {
        self.add_op([], Ok::<_, IRError>(Input(ty))).expect("Constructing leaf is infallible!")[0]
    }

    #[must_use]
    pub fn add_const(&mut self, value: TValue) -> NodeId {
        self.add_op([], Ok::<_, IRError>(Constant(value))).expect("Constructing leaf is infallible!")[0]
    }

    pub fn add_scalar(&mut self, value: impl Into<DValue>, size: impl Into<Size>) -> NodeId {
        self.add_op([], Ok::<_, IRError>(ScalarConstant(value.into(), size.into())))
            .expect("Constructing leaf is infallible!")[0]
    }

    pub fn add_broadcast(
        &mut self,
        input: NodeId,
        shape: impl Into<Shape>,
        dim: usize,
        repeats: impl Into<Size>,
    ) -> Result<NodeId, IRTrace> {
        let dtype = self.get_node(input)?.ty().dtype();
        let broadcast = BroadcastAcrossDimension::new(dtype, shape.into(), dim, repeats.into());
        self.add_op([input], broadcast).map(|x| x[0])
    }

    pub fn add_unary(&mut self, node: NodeId, op: Unary) -> Result<NodeId, IRTrace> {
        let op = self.get_node(node).and_then(|node| UnaryOp::new(node.ty(), op).map_err(IRTrace::Root));
        self.add_op([node], op).map(|x| x[0])
    }

    pub fn add_binary(&mut self, lhs: NodeId, rhs: NodeId, op: CABinary) -> Result<NodeId, IRTrace> {
        let ty = self.get_node(lhs)?.ty();
        if ty != self.get_node(rhs)?.ty() {
            return Err(format!("Mismatched input types to CABinary::{op:?}").into());
        }

        let op = CABinaryOp::new(ty, op);
        self.add_op([lhs, rhs], Ok::<_, IRTrace>(op)).map(|x| x[0])
    }

    pub fn copy(&mut self, node: NodeId) -> Result<NodeId, IRTrace> {
        self.add_op([node], self.get_node(node).map(|n| CopyOp(n.ty()))).map(|x| x[0])
    }

    pub fn eliminate_dead_ops(&mut self) -> Result<(), IRTrace> {
        self.transform(EliminateCopies)?;
        self.transform(EliminateUnusedOperations)
    }

    pub fn swap_outputs(&mut self, id1: NodeId, id2: NodeId) -> Result<(), IRTrace> {
        self.transform(SwapOutputs(id1, id2))
    }

    pub fn remove_op(&mut self, id: OpId) -> Result<(), IRTrace> {
        self.transform(RemoveOperation(id))
    }

    pub fn replace_input(&mut self, new: NodeId, old: NodeId) -> Result<(), IRTrace> {
        self.transform(ReplaceInput { new, old })
    }

    pub fn replace_op(&mut self, op: OpId, new: AddOperation) -> Result<OpId, IRTrace> {
        let first_output = self.get_op(op)?.outputs()[0];
        self.transform(ReplaceOperation(op, new))?;
        self.get_parent_op(first_output)
    }

    pub fn replace_operation(
        &mut self,
        op: OpId,
        new_inputs: impl Into<Vec<NodeId>>,
        new_op: impl OpType,
    ) -> Result<OpId, IRTrace> {
        let add = AddOperation::new(new_inputs.into(), Ok(Rc::new(new_op)));
        self.replace_op(op, add)
    }

    pub fn get_dependent_ops_set(&self, op: OpId) -> Result<HashSet<OpId>, IRTrace> {
        self.ir.get_dependent_ops_set(op).map_err(|e| e.into())
    }

    pub fn optimise(&mut self) -> Result<(), IRTrace> {
        self.transform(CanonicalisePass::all())
    }
}

#[derive(Clone)]
pub enum IRTrace {
    Root(IRError),
    Frame(Box<IR<Tensor>>, Rc<dyn IRTransform>, Rc<Self>),
}

impl IRTrace {
    pub fn frame(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::Root(err) => write!(f, "{err:?}"),
            Self::Frame(ir, transform, _) => {
                let orange = Ansi::rgb(212, 114, 34);
                let clear = Ansi::Clear;

                writeln!(f, "{orange}Error applying{clear}")?;
                writeln!(f, "{transform:?}")?;
                writeln!(f, "{orange}on ir{clear}")?;
                write!(f, "{ir}")
            }
        }
    }

    pub fn full_string(&self, f: &mut impl fmt::Write, frame: usize) -> fmt::Result {
        writeln!(f, "{}Depth {frame}:{}", Ansi::rgb(255, 0, 0), Ansi::Clear)?;

        self.frame(f)?;

        if let Self::Frame(_, _, inner) = self {
            writeln!(f)?;
            inner.full_string(f, frame + 1)?;
        }

        Ok(())
    }
}

impl From<IRError> for IRTrace {
    fn from(value: IRError) -> Self {
        Self::Root(value)
    }
}

impl<T: Into<String>> From<T> for IRTrace {
    fn from(value: T) -> Self {
        Self::Root(value.into().into())
    }
}

impl fmt::Debug for IRTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.full_string(f, 0)
    }
}

impl fmt::Display for TensorIR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn map<T>(x: Result<T, IRTrace>) -> Result<T, fmt::Error> {
            x.map_err(|_| fmt::Error)
        }

        write!(f, "irgraph(")?;
        let leaves = self.ir.operations().filter(|x| x.data().is_input()).collect::<Vec<_>>();
        let mline = leaves.len() >= 5;

        for (i, leaf) in leaves.iter().enumerate() {
            if mline {
                writeln!(f)?;
                write!(f, "    ")?;
            } else if i != 0 {
                write!(f, ", ")?;
            }

            let node = leaf.outputs()[0];
            let ty = map(self.get_node(node))?.ty();

            write!(f, "{node:?}: {ty:?}")?;
        }

        if mline {
            writeln!(f)?;
        }

        writeln!(f, ") {{")?;

        for op in map(self.ordered_operations())? {
            if op.data().is_input() {
                continue;
            }

            let inputs = op.inputs();
            let outputs = op.outputs();

            write!(f, "    ")?;
            if outputs.len() > 1 {
                write!(f, "[")?;
            }

            let output_tys =
                map(outputs.iter().map(|x| self.get_node(*x).map(Node::ty)).collect::<Result<Vec<_>, _>>())?;

            for (i, (&output, ty)) in outputs.iter().zip(output_tys).enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }

                write!(f, "{output:?}: {ty:?}")?;
            }

            if outputs.len() > 1 {
                write!(f, "]")?;
            }

            write!(f, " = {}(", op.data().0.opname())?;

            for (i, &input) in inputs.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }

                write!(f, "{input:?}")?;
            }

            writeln!(f, ")")?;
        }

        write!(f, "    return ")?;
        for (i, &output) in self.outputs.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }

            write!(f, "{output:?}")?;
        }

        writeln!(f)?;
        write!(f, "}}")?;

        Ok(())
    }
}
