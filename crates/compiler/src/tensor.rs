mod builder;
mod highlight;
mod history;
pub mod operation;
mod pattern;
pub mod transform;
mod ttype;

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fmt,
    rc::Rc,
};

use crate::ir::{IR, IRError, Node, NodeId, Op, OpId, TypeSystem};

use highlight::Emit;
use operation::{
    BroadcastAcrossDimension, CABinary, CABinaryOp, Constant, CopyOp, Input, ReduceAcrossDimension, Reduction,
    ScalarConstant, SubGraph, Unary, UnaryOp,
};
use transform::{IRTransform, canonicalise, eliminate, modify};

pub use builder::{IRBuilder, TNode};
pub use half::{bf16, f16};
pub use highlight::Token;
pub use history::{Frame, IRHistory, Line};
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
    outputs: BTreeSet<NodeId>,
    /// Run `check_valid` after every edit
    /// Useful for debugging compiler passes, slows compilation
    pub debug: bool,
    /// Records the IR after each core edit while recording, see `start_recording`
    history: IRHistory,
}

impl TensorIR {
    pub fn ir(&self) -> &IR<Tensor> {
        &self.ir
    }

    pub fn ir_mut(&mut self) -> &mut IR<Tensor> {
        &mut self.ir
    }

    pub fn evaluate(
        &self,
        inputs: impl Into<BTreeMap<NodeId, TValue>>,
    ) -> Result<Option<BTreeMap<NodeId, TValue>>, IRTrace> {
        let mut values: BTreeMap<_, _> =
            inputs.into().into_iter().map(|(id, tensor)| (id, RefCell::new(tensor))).collect();

        for (id, tensor) in &values {
            let op = self.get_op(self.get_parent_op(*id)?)?;
            if !op.data().is_input() {
                return Err("Seeded non-leaf node!".into());
            }

            assert_eq!(tensor.borrow().size(), self.get_node(*id)?.ty().size().get());
        }

        for op in self.ordered_operations()? {
            for &output in op.outputs() {
                let ty = self.get_node(output)?.ty();
                let size = ty.size().get();
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

            // inputs are seeded, anything else must be evaluable
            if !op.data().is_input()
                && !op.data().0.evaluate(
                    op_inputs.iter().map(|x| &**x).collect(),
                    op_outputs.iter_mut().map(|x| &mut **x).collect(),
                )
            {
                return Ok(None);
            }
        }

        let filter = |x: (_, RefCell<_>)| self.is_output(x.0).then(|| (x.0, x.1.into_inner()));
        Ok(Some(values.into_iter().filter_map(filter).collect()))
    }

    pub fn transform(&mut self, transform: impl IRTransform) -> Result<(), IRTrace> {
        self.transform_dyn(Rc::new(transform))
    }

    pub fn transform_dyn(&mut self, transform: Rc<dyn IRTransform>) -> Result<(), IRTrace> {
        let ir = Box::new(self.ir().clone());
        let result = self.labelled(transform.type_name(), |ir| transform.apply(ir));
        result.map_err(|err| IRTrace::Frame(ir, transform, Rc::new(err)))
    }

    /// Run `f` as a pass named by `type_name` in the `IRHistory`, without the
    /// IR snapshot that `transform` takes for error traces, for use in hot loops.
    pub(crate) fn labelled<T>(&mut self, type_name: &'static str, f: impl FnOnce(&mut Self) -> T) -> T {
        self.history.push_pass(type_name);
        let result = f(self);
        self.history.pop_pass();
        result
    }

    pub fn ordered_operations(&self) -> Result<Vec<Op<Tensor>>, IRTrace> {
        self.ir.ordered_operations().map_err(IRTrace::Root)
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

    fn debug_check_edit(&self) -> Result<(), IRTrace> {
        if self.debug { self.check_valid() } else { Ok(()) }
    }

    pub fn history(&self) -> &IRHistory {
        &self.history
    }

    /// Start recording edits to the `IRHistory`, which is stopped by default.
    /// Records a frame of the current IR labelled with `label`, covering anything
    /// that happened while stopped.
    /// Does nothing if already recording.
    pub fn start_recording(&mut self, label: impl Into<String>) {
        if !self.history.is_recording() {
            let ir = history::display(self);
            self.history.start(ir, label.into());
        }
    }

    /// Stop recording edits to the `IRHistory`, see `start_recording`.
    pub fn stop_recording(&mut self) {
        self.history.stop();
    }

    fn record_edit(&mut self, edit: &dyn IRTransform) {
        if self.history.is_recording() {
            let ir = history::display(self);
            self.history.record(ir, edit);
        }
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
        op: Result<TensorOp, IRTrace>,
    ) -> Result<Vec<NodeId>, IRTrace> {
        let inputs = inputs.as_ref().to_vec();
        let transform = modify::AddOperation::new(inputs, op);
        self.transform(transform.clone()).map(|_| transform.outputs())
    }

    pub fn add_op(
        &mut self,
        inputs: impl AsRef<[NodeId]>,
        op: Result<impl OpType, impl Into<IRTrace>>,
    ) -> Result<Vec<NodeId>, IRTrace> {
        fn convert(x: impl OpType) -> TensorOp {
            TensorOp(Rc::new(x))
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

    pub fn add_reduction(
        &mut self,
        input: NodeId,
        shape: impl Into<Shape>,
        dim: usize,
        reduction: Reduction,
    ) -> Result<NodeId, IRTrace> {
        let dtype = self.get_node(input)?.ty().dtype();
        let reduce = ReduceAcrossDimension::new(dtype, shape.into(), dim, reduction);
        self.add_op([input], reduce).map(|x| x[0])
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
        self.transform(eliminate::EliminateCopies)?;
        self.transform(eliminate::EliminateUnusedOperations)
    }

    pub fn swap_outputs(&mut self, id1: NodeId, id2: NodeId) -> Result<(), IRTrace> {
        self.transform(modify::SwapOutputs(id1, id2))
    }

    pub fn remove_op(&mut self, id: OpId) -> Result<(), IRTrace> {
        self.transform(modify::RemoveOperation(id))
    }

    pub fn replace_input(&mut self, new: NodeId, old: NodeId) -> Result<(), IRTrace> {
        self.transform(modify::ReplaceInput { new, old })
    }

    pub fn replace_op(&mut self, op: OpId, new: modify::AddOperation) -> Result<OpId, IRTrace> {
        let first_output = self.get_op(op)?.outputs()[0];
        self.transform(modify::ReplaceOperation(op, new))?;
        self.get_parent_op(first_output)
    }

    pub fn replace_operation(
        &mut self,
        op: OpId,
        new_inputs: impl Into<Vec<NodeId>>,
        new_op: impl OpType,
    ) -> Result<OpId, IRTrace> {
        let add = modify::AddOperation::new(new_inputs.into(), Ok(TensorOp::new(new_op)));
        self.replace_op(op, add)
    }

    pub fn get_dependent_ops_set(&self, op: OpId) -> Result<BTreeSet<OpId>, IRTrace> {
        self.ir.get_dependent_ops_set(op).map_err(|e| e.into())
    }

    pub fn is_dependent_op(&self, id: OpId, target: OpId) -> Result<bool, IRTrace> {
        self.ir.is_dependent_op(id, target).map_err(|e| e.into())
    }

    pub fn is_immediate_dependent_op(&self, parent: OpId, child: OpId) -> Result<bool, IRTrace> {
        self.ir.is_immediate_dependent_op(parent, child).map_err(|e| e.into())
    }

    pub fn optimise(&mut self) -> Result<(), IRTrace> {
        self.transform(canonicalise::CanonicalisePass::all())
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
                writeln!(f, "Error applying")?;
                writeln!(f, "{transform:?}")?;
                writeln!(f, "on ir")?;
                write!(f, "{ir}")
            }
        }
    }

    pub fn full_string(&self, f: &mut impl fmt::Write, frame: usize) -> fmt::Result {
        writeln!(f, "Depth {frame}:")?;

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

impl fmt::Display for IRTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for IRTrace {}

impl TensorIR {
    /// Write the display form of the IR as a sequence of tokens, with their syntax class if they have one.
    pub fn write_ir(&self, emit: &mut dyn FnMut(Option<Token>, &str) -> fmt::Result) -> fmt::Result {
        emit(Some(Token::Keyword), "irgraph")?;
        emit(Some(Token::Bracket(0)), "(")?;
        let leaves = self.ir.operations().filter(|x| x.data().is_input()).map(|x| x.outputs()[0]).collect::<Vec<_>>();
        let mline = leaves.len() >= 5;

        for (i, &leaf) in leaves.iter().enumerate() {
            if mline {
                emit(None, "\n    ")?;
            } else if i != 0 {
                emit(None, ", ")?;
            }

            self.write_param(leaf, 0, emit)?;
        }

        if mline {
            emit(None, "\n")?;
        }

        emit(Some(Token::Bracket(0)), ")")?;
        emit(None, " ")?;
        self.write_body(&self.outputs.iter().copied().collect::<Vec<_>>(), 0, emit)
    }

    /// Name of `node` in a graph nested `level` subgraphs deep, prefixed so that it
    /// does not shadow the names of the enclosing graphs.
    fn node_name(node: NodeId, level: usize) -> String {
        format!("%{}{}", "s".repeat(level), node.inner())
    }

    fn write_param(&self, node: NodeId, level: usize, emit: Emit<'_>) -> fmt::Result {
        let ty = self.get_node(node).map_err(|_| fmt::Error)?.ty();
        emit(None, &Self::node_name(node, level))?;
        emit(Some(Token::Punct), ":")?;
        emit(None, " ")?;
        emit(Some(Token::Type), &format!("{:?}", ty.dtype()))?;
        emit(Some(Token::Bracket(level + 1)), "[")?;
        emit(Some(Token::Number), &format!("{:?}", ty.size()))?;
        emit(Some(Token::Bracket(level + 1)), "]")
    }

    /// If `subgraph` is a single operation, which is not itself displayed as a closure, and returns
    /// exactly its outputs, the operation's name and its inputs in terms of the `outer` inputs.
    fn single_op(subgraph: &SubGraph, outer: &[NodeId]) -> Option<(String, Vec<NodeId>)> {
        let mut ops = subgraph.internal_graph().ir.operations().filter(|op| !op.data().is_input());
        let op = ops.next()?;

        if ops.next().is_some() || op.data().0.as_closure().is_some() || op.outputs() != subgraph.internal_outputs() {
            return None;
        }

        let inputs = subgraph.internal_inputs();
        let outer = op.inputs().iter().map(|node| inputs.iter().position(|x| x == node).map(|i| outer[i]));
        Some((op.data().0.opname(), outer.collect::<Option<_>>()?))
    }

    /// Write `{ ...operations, return outputs }`, with the operations indented one level
    /// further than `level`, and subgraphs written as closures after their call.
    fn write_body(&self, outputs: &[NodeId], level: usize, emit: Emit<'_>) -> fmt::Result {
        let indent = "    ".repeat(level + 1);
        let list = |nodes: &[NodeId], emit: Emit<'_>| {
            comma_separated(nodes, emit, |node, emit| emit(None, &Self::node_name(node, level)))
        };

        emit(Some(Token::Bracket(level)), "{")?;
        emit(None, "\n")?;

        for op in self.ordered_operations().map_err(|_| fmt::Error)? {
            if op.data().is_input() {
                continue;
            }

            let outputs = op.outputs();

            emit(None, &indent)?;
            if outputs.len() > 1 {
                emit(Some(Token::Bracket(level + 1)), "[")?;
            }

            list(outputs, emit)?;

            if outputs.len() > 1 {
                emit(Some(Token::Bracket(level + 1)), "]")?;
            }

            emit(None, " ")?;
            emit(Some(Token::Punct), "=")?;
            emit(None, " ")?;

            let (opname, inputs, closure) = match op.data().0.as_closure() {
                Some((name, subgraph)) => match Self::single_op(subgraph, op.inputs()) {
                    // a single operation is written inline, e.g. `fused.binary.add(%1, %2)`
                    Some((inner, inputs)) => (format!("{name}.{inner}"), inputs, None),
                    None => (name.to_string(), op.inputs().to_vec(), Some(subgraph)),
                },
                None => (op.data().0.opname(), op.inputs().to_vec(), None),
            };

            highlight::opname(&opname, level + 1, emit)?;
            emit(Some(Token::Bracket(level + 1)), "(")?;
            list(&inputs, emit)?;
            emit(Some(Token::Bracket(level + 1)), ")")?;

            if let Some(subgraph) = closure {
                let graph = subgraph.internal_graph();
                emit(None, " ")?;
                emit(Some(Token::Punct), "|")?;
                comma_separated(subgraph.internal_inputs(), emit, |input, emit| {
                    graph.write_param(input, level + 1, emit)
                })?;
                emit(Some(Token::Punct), "|")?;
                emit(None, " ")?;
                graph.write_body(subgraph.internal_outputs(), level + 1, emit)?;
            }

            emit(None, "\n")?;
        }

        emit(None, &indent)?;
        emit(Some(Token::Keyword), "return")?;
        emit(None, " ")?;
        list(outputs, emit)?;
        emit(None, "\n")?;
        emit(None, &"    ".repeat(level))?;
        emit(Some(Token::Bracket(level)), "}")
    }
}

/// Write each of `items` with `write`, separated by commas.
fn comma_separated<T: Copy>(items: &[T], emit: Emit<'_>, write: impl Fn(T, Emit<'_>) -> fmt::Result) -> fmt::Result {
    for (i, &item) in items.iter().enumerate() {
        if i != 0 {
            emit(None, ", ")?;
        }

        write(item, emit)?;
    }

    Ok(())
}

impl fmt::Display for TensorIR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_ir(&mut |_, text| f.write_str(text))
    }
}
