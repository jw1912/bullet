pub mod frontend;
pub mod graph;
pub mod operation;
pub mod transform;
pub mod utils;

pub mod prelude {
    pub use crate::{
        frontend::{IRBuilder, IRNode},
        graph::{DType, DValue, Shape, Size, TType, TValue},
    };
}

use std::{
    collections::{HashMap, HashSet},
    fmt,
    rc::Rc,
};

use graph::{DValue, Graph, GraphError, Input, Node, NodeId, Op, OpId, OpType, Shape, Size, TType, TValue};
use operation::{BroadcastAcrossDimension, CABinary, CABinaryOp, Constant, CopyOp, ScalarConstant, Unary, UnaryOp};
use transform::{
    CanonicalisePass, IRTransform,
    eliminate::{EliminateCopies, EliminateUnusedOperations},
    modify::{AddOperation, RemoveOperation, ReplaceInput, ReplaceOperation, SwapOutputs},
};
use utils::Ansi;

#[derive(Clone, Default)]
pub struct IR {
    graph: Graph,
    history: Option<IRHistory>,
}

impl fmt::Display for IR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Current IR Graph:")?;
        writeln!(f, "{}", self.graph.as_highlighted())?;
        if let Some(history) = &self.history {
            write!(f, "{history}")?;
        }

        Ok(())
    }
}

impl IR {
    pub fn graph(&self) -> Graph {
        self.graph.clone()
    }

    pub fn evaluate(&self, inputs: impl Into<HashMap<NodeId, TValue>>) -> Result<HashMap<NodeId, TValue>, GraphError> {
        self.graph.evaluate(inputs)
    }

    pub fn transform(&mut self, transform: impl IRTransform) -> Result<(), IRTrace> {
        self.transform_dyn(Rc::new(transform))
    }

    pub fn transform_dyn(&mut self, transform: Rc<dyn IRTransform>) -> Result<(), IRTrace> {
        let graph = Box::new(self.graph());

        if let Some(history) = &mut self.history {
            history.push(transform.clone());
            history.start_scope();
        }

        transform.apply(self).map_err(|err| IRTrace::Frame(graph, transform, Rc::new(err)))?;

        if let Some(history) = &mut self.history {
            history.end_scope();
        }

        Ok(())
    }

    pub fn ordered_operations(&self) -> Result<Vec<Op>, IRTrace> {
        let ids = self.graph.topo_order_ops()?;
        ids.into_iter().map(|id| self.graph.get_op(id).map_err(IRTrace::Root).cloned()).collect()
    }

    pub fn operations(&self) -> Vec<Op> {
        self.graph.operations().cloned().collect()
    }

    pub fn num_nontrivial_operations(&self) -> Result<usize, IRTrace> {
        let mut count = 0;

        for op in self.ordered_operations()? {
            if !op.is_input() && op.downcast::<Constant>().is_none() && op.downcast::<ScalarConstant>().is_none() {
                count += 1;
            }
        }

        Ok(count)
    }

    pub fn num_ops(&self) -> usize {
        self.graph.num_ops()
    }

    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    pub fn get_parent_op(&self, node: NodeId) -> Result<OpId, IRTrace> {
        self.graph.get_parent_op(node).map_err(IRTrace::Root)
    }

    pub fn get_op(&self, op: OpId) -> Result<&Op, IRTrace> {
        self.graph.get_op(op).map_err(IRTrace::Root)
    }

    pub fn get_op_mut(&mut self, op: OpId) -> Result<&mut Op, IRTrace> {
        self.graph.get_op_mut(op).map_err(IRTrace::Root)
    }

    pub fn get_node(&self, node: NodeId) -> Result<&Node, IRTrace> {
        self.graph.get_node(node).map_err(IRTrace::Root)
    }

    pub fn get_node_mut(&mut self, node: NodeId) -> Result<&mut Node, IRTrace> {
        self.graph.get_node_mut(node).map_err(IRTrace::Root)
    }

    pub fn is_output(&self, node: NodeId) -> bool {
        self.graph.is_output(node)
    }

    pub fn is_copy(&self, node: NodeId) -> Result<Option<NodeId>, IRTrace> {
        let op = self.get_op(self.get_parent_op(node)?)?;
        Ok(op.downcast::<CopyOp>().map(|_| op.inputs()[0]))
    }

    pub fn are_copies(&self, a: NodeId, b: NodeId) -> Result<bool, IRTrace> {
        Ok(self.is_copy(a)? == Some(b) || self.is_copy(b)? == Some(a))
    }

    pub fn parent_op<T: OpType>(&self, node: NodeId) -> Result<Option<&T>, IRTrace> {
        let id = self.get_parent_op(node)?;
        let op = self.get_op(id)?;
        Ok(op.downcast::<T>())
    }

    pub fn is_input(&self, node: NodeId) -> Result<bool, IRTrace> {
        self.parent_op::<Input>(node).map(|x| x.is_some())
    }

    pub fn is_constant(&self, node: NodeId) -> Result<bool, IRTrace> {
        Ok(self.parent_op::<Constant>(node)?.is_some() || self.parent_op::<ScalarConstant>(node)?.is_some())
    }

    pub fn check_valid(&self) -> Result<(), IRTrace> {
        self.graph.check_valid().map_err(IRTrace::Root)
    }

    pub fn register_output(&mut self, node: NodeId) {
        self.graph.register_output(node);
    }

    pub fn unregister_output(&mut self, node: NodeId) {
        self.graph.register_output(node);
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
        self.add_op([], Ok::<_, GraphError>(Input(ty))).expect("Constructing leaf is infallible!")[0]
    }

    #[must_use]
    pub fn add_const(&mut self, value: TValue) -> NodeId {
        self.add_op([], Ok::<_, GraphError>(Constant(value))).expect("Constructing leaf is infallible!")[0]
    }

    pub fn add_scalar(&mut self, value: impl Into<DValue>, size: impl Into<Size>) -> NodeId {
        self.add_op([], Ok::<_, GraphError>(ScalarConstant(value.into(), size.into())))
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
        self.graph.get_dependent_ops_set(op).map_err(|e| e.into())
    }

    pub fn optimise(&mut self) -> Result<(), IRTrace> {
        self.transform(CanonicalisePass::all())
    }

    pub fn track_history(&mut self) {
        self.history = Some(IRHistory::default());
    }

    pub fn untrack_history(&mut self) {
        self.history = None;
    }
}

#[derive(Clone)]
pub enum IRTrace {
    Root(GraphError),
    Frame(Box<Graph>, Rc<dyn IRTransform>, Rc<Self>),
}

impl IRTrace {
    pub fn frame(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::Root(err) => write!(f, "{err:?}"),
            Self::Frame(graph, transform, _) => {
                let orange = Ansi::rgb(212, 114, 34);
                let clear = Ansi::Clear;

                writeln!(f, "{orange}Error applying{clear}")?;
                writeln!(f, "{transform:?}")?;
                writeln!(f, "{orange}on graph{clear}")?;
                write!(f, "{}", graph.as_highlighted())
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

impl From<GraphError> for IRTrace {
    fn from(value: GraphError) -> Self {
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

#[derive(Clone, Debug)]
enum IRHistoryFrame {
    ScopeStart,
    ScopeEnd,
    Entry(Rc<dyn IRTransform>),
}

#[derive(Clone, Debug, Default)]
pub struct IRHistory(Vec<IRHistoryFrame>);

impl fmt::Display for IRHistory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut scope = 0;

        write!(f, "IR Transform History")?;

        for entry in &self.0 {
            match entry {
                IRHistoryFrame::ScopeStart => scope += 1,
                IRHistoryFrame::ScopeEnd => scope -= 1,
                IRHistoryFrame::Entry(entry) => {
                    writeln!(f)?;
                    write!(f, "{}|-- {entry:?}", " ".repeat(4 * scope))?;
                }
            }
        }

        Ok(())
    }
}

impl IRHistory {
    pub fn start_scope(&mut self) {
        self.0.push(IRHistoryFrame::ScopeStart);
    }

    pub fn end_scope(&mut self) {
        self.0.push(IRHistoryFrame::ScopeEnd);
    }

    pub fn push(&mut self, transform: Rc<dyn IRTransform>) {
        self.0.push(IRHistoryFrame::Entry(transform));
    }
}
