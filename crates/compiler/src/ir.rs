pub mod graph;
pub mod transform;

use std::{collections::HashMap, fmt, rc::Rc};

use graph::{
    IrError, IrGraph, IrNode, IrNodeId, IrOperation, IrOperationId, IrOperationType, IrType,
    operation::{Constant, FusedElementwise, IrBinary, IrCopy, IrUnary, Leaf},
};
use transform::*;

use crate::{
    core::{Binary, DTypeTensor, Formula, FormulaId, Unary},
    utils::Ansi,
};

#[derive(Clone, Default)]
pub struct IR {
    graph: IrGraph,
    most_recent: Vec<IrNodeId>,
}

#[derive(Clone)]
pub enum IRTrace {
    Root(IrError),
    Frame(Box<IrGraph>, Rc<dyn IrTransform>, Rc<Self>),
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

impl From<IrError> for IRTrace {
    fn from(value: IrError) -> Self {
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

impl IR {
    pub fn graph(&self) -> IrGraph {
        self.graph.clone()
    }

    pub fn as_highlighted(&self) -> String {
        self.graph.as_highlighted()
    }

    pub fn evaluate(
        &self,
        inputs: impl Into<HashMap<IrNodeId, DTypeTensor>>,
    ) -> Result<HashMap<IrNodeId, DTypeTensor>, IrError> {
        self.graph.evaluate(inputs)
    }

    pub fn transform(&mut self, transform: impl IrTransform) -> Result<(), IRTrace> {
        let graph = Box::new(self.graph());
        transform.apply(self).map_err(|err| IRTrace::Frame(graph, Rc::new(transform), Rc::new(err)))
    }

    pub fn ordered_operations(&self) -> Result<Vec<IrOperation>, IRTrace> {
        let ids = self.graph.topo_order_ops()?;
        ids.into_iter().map(|id| self.graph.get_op(id).map_err(IRTrace::Root).cloned()).collect()
    }

    pub fn operations(&self) -> Vec<IrOperation> {
        self.graph.operations().cloned().collect()
    }

    pub fn num_nontrivial_operations(&self) -> Result<usize, IRTrace> {
        let mut count = 0;

        for op in self.ordered_operations()? {
            if IrOperation::downcast::<Leaf>(op.op()).is_none() && IrOperation::downcast::<Constant>(op.op()).is_none()
            {
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

    pub fn get_parent_op(&self, node: IrNodeId) -> Result<IrOperationId, IRTrace> {
        self.graph.get_parent_op(node).map_err(IRTrace::Root)
    }

    pub fn get_op(&self, op: IrOperationId) -> Result<&IrOperation, IRTrace> {
        self.graph.get_op(op).map_err(IRTrace::Root)
    }

    pub fn get_op_mut(&mut self, op: IrOperationId) -> Result<&mut IrOperation, IRTrace> {
        self.graph.get_op_mut(op).map_err(IRTrace::Root)
    }

    pub fn get_node(&self, node: IrNodeId) -> Result<&IrNode, IRTrace> {
        self.graph.get_node(node).map_err(IRTrace::Root)
    }

    pub fn get_node_mut(&mut self, node: IrNodeId) -> Result<&mut IrNode, IRTrace> {
        self.graph.get_node_mut(node).map_err(IRTrace::Root)
    }

    pub fn is_output(&self, node: IrNodeId) -> bool {
        self.graph.is_output(node)
    }

    pub fn is_copy(&self, node: IrNodeId) -> Result<Option<IrNodeId>, IRTrace> {
        let op = self.get_op(self.get_parent_op(node)?)?;
        Ok(IrOperation::downcast::<IrCopy>(op.op()).is_some().then(|| op.inputs()[0]))
    }

    pub fn is_child_of<T: IrOperationType>(&self, node: IrNodeId) -> Result<bool, IRTrace> {
        let id = self.get_parent_op(node)?;
        let op = self.get_op(id)?;
        Ok(IrOperation::downcast::<T>(op.op()).is_some())
    }

    pub fn is_input(&self, node: IrNodeId) -> Result<bool, IRTrace> {
        self.is_child_of::<Leaf>(node)
    }

    pub fn is_constant(&self, node: IrNodeId) -> Result<bool, IRTrace> {
        self.is_child_of::<Constant>(node)
    }

    pub fn check_valid(&self) -> Result<(), IRTrace> {
        self.graph.check_valid().map_err(IRTrace::Root)
    }

    pub fn register_output(&mut self, node: IrNodeId) {
        self.graph.register_output(node);
    }

    pub fn unregister_output(&mut self, node: IrNodeId) {
        self.graph.register_output(node);
    }

    pub fn add_op(
        &mut self,
        inputs: impl AsRef<[IrNodeId]>,
        op: Result<impl IrOperationType, impl Into<IRTrace>>,
    ) -> Result<Vec<IrNodeId>, IRTrace> {
        fn convert(x: impl IrOperationType) -> Rc<dyn IrOperationType> {
            Rc::new(x)
        }

        let inputs = inputs.as_ref().to_vec();
        let op = op.map(convert).map_err(|e| e.into());

        self.transform(AddOperation(inputs, op)).map(|_| self.most_recent.clone())
    }

    #[must_use]
    pub fn add_leaf(&mut self, ty: IrType) -> IrNodeId {
        self.add_op([], Ok::<_, IrError>(Leaf(ty))).expect("Constructing leaf is infallible!")[0]
    }

    #[must_use]
    pub fn add_const(&mut self, value: DTypeTensor) -> IrNodeId {
        self.add_op([], Ok::<_, IrError>(Constant(value))).expect("Constructing leaf is infallible!")[0]
    }

    pub fn add_unary(&mut self, node: IrNodeId, op: Unary) -> Result<IrNodeId, IRTrace> {
        let op = self.get_node(node).and_then(|node| IrUnary::new(node.ty(), op).map_err(IRTrace::Root));
        self.add_op([node], op).map(|x| x[0])
    }

    pub fn add_binary(&mut self, lhs: IrNodeId, rhs: IrNodeId, op: Binary) -> Result<IrNodeId, IRTrace> {
        let op = IrBinary::new(self.get_node(lhs)?.ty(), self.get_node(rhs)?.ty(), op);
        self.add_op([lhs, rhs], op).map(|x| x[0])
    }

    pub fn copy(&mut self, node: IrNodeId) -> Result<IrNodeId, IRTrace> {
        self.add_op([node], self.get_node(node).map(|n| IrCopy(n.ty()))).map(|x| x[0])
    }

    pub fn add_elementwise<const M: usize, const N: usize, F>(
        &mut self,
        inputs: [IrNodeId; M],
        f: F,
    ) -> Result<[IrNodeId; N], IRTrace>
    where
        F: for<'a> Fn(&mut Formula, [FormulaId; M]) -> Option<[FormulaId; N]>,
    {
        let nodes = inputs.map(|x| self.get_node(x).unwrap().ty());
        let op = FusedElementwise::new(nodes, f);

        let outs = self.add_op(inputs, op)?;

        let mut output = [outs[0]; N];

        for (i, j) in output.iter_mut().zip(outs) {
            *i = j;
        }

        Ok(output)
    }

    pub fn swap_outputs(&mut self, id1: IrNodeId, id2: IrNodeId) -> Result<(), IRTrace> {
        self.transform(SwapOutputs(id1, id2))
    }

    pub fn remove_op(&mut self, id: IrOperationId) -> Result<(), IRTrace> {
        self.transform(RemoveOperation(id))
    }

    pub fn replace_input(&mut self, new: IrNodeId, old: IrNodeId) -> Result<(), IRTrace> {
        self.transform(ReplaceInput { new, old })
    }

    pub fn optimise(&mut self) -> Result<(), IRTrace> {
        self.transform(DecomposeElementwise)?;
        self.transform(EliminateUnusedOperations)?;
        self.transform(FoldBroadcasts)?;
        self.transform(FoldConstants)?;
        self.transform(CanonicaliseInputs)?;
        self.transform(EliminateCommonSubExpressions)?;
        self.transform(EliminateCopies)
    }
}
