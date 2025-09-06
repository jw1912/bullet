pub mod compile;
pub mod node;
pub mod operation;
pub mod passes;

use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug},
    marker::PhantomData,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use node::{AnnotatedNode, NodeInfo};
use operation::*;

use crate::{
    dag::{
        DAGraph, DAGraphError, Node, NodeId, Operation,
        format::StringOperation,
        manager::{DAGraphManager, DAGraphManagerError, DAGraphType},
    },
    device::{Device, tensor::Shape},
};

pub trait BackendMarker: Copy + Debug + Default + 'static {
    type Backend: Device;
}

#[derive(Default)]
pub struct GraphIRManager<B: BackendMarker> {
    inner: DAGraphManager<GraphIRType<B>>,
    inputs: HashSet<NodeId>,
    weights: HashSet<NodeId>,
    ids: HashMap<NodeId, String>,
}

impl<B: BackendMarker> GraphIRManager<B> {
    pub fn get_id(&self, id: NodeId) -> Option<String> {
        self.ids.get(&id).cloned()
    }

    pub fn get(&self, id: NodeId) -> GraphIRResult<&Node<NodeInfo, GraphIROperation<B>>, B> {
        self.inner.get(id)
    }

    pub fn modify<T>(&mut self, f: impl FnOnce(&mut GraphIR<B>) -> Result<T, DAGraphError>) -> GraphIRResult<T, B> {
        self.inner.modify(f)
    }

    pub fn root(&self) -> GraphIRResult<AnnotatedNode, B> {
        let roots = self.inner.roots();

        if roots.len() != 1 {
            return self.inner.capture_error(Err(GraphIRError::MultipleRoots.into()));
        }

        let idx = *roots.iter().next().unwrap();
        let shape = self.get(idx)?.ty().shape;

        Ok(AnnotatedNode { idx, shape })
    }

    pub fn formatted(&self) -> Result<DAGraph<String, StringOperation>, DAGraphError> {
        self.inner.formatted()
    }

    pub fn add_leaf(
        &mut self,
        id: Option<String>,
        shape: Shape,
        batched: bool,
        requires_grad: bool,
        sparse: Option<NonZeroUsize>,
    ) -> Result<AnnotatedNode, DAGraphManagerError<GraphIRType<B>>> {
        let ty = NodeInfo { shape, batched, requires_grad, sparse };
        let node = self.add_op(GraphIRLeaf { id: id.clone(), ty })?;

        if let Some(id) = id {
            if self.ids.insert(node.idx, id.clone()).is_some() {
                let err = Err(GraphIRError::NodeWithIdAlreadyExists(id));
                return self.inner.capture_error(err.map_err(Into::into))?;
            }
        }

        Ok(node)
    }

    pub fn add_constant(&mut self, shape: Shape) -> GraphIRResult<AnnotatedNode, B> {
        let node = self.add_leaf(None, shape, false, false, None)?;
        Ok(node)
    }

    pub fn add_dense_input(&mut self, id: &str, shape: Shape) -> GraphIRResult<AnnotatedNode, B> {
        let node = self.add_leaf(Some(id.to_string()), shape, true, false, None)?;
        self.inputs.insert(node.idx);
        Ok(node)
    }

    pub fn add_sparse_input(&mut self, id: &str, shape: Shape, nnz: usize) -> GraphIRResult<AnnotatedNode, B> {
        let nnz = NonZeroUsize::try_from(nnz).unwrap();
        let node = self.add_leaf(Some(id.to_string()), shape, true, false, Some(nnz))?;
        self.inputs.insert(node.idx);
        Ok(node)
    }

    pub fn add_unbatched_input(
        &mut self,
        id: &str,
        shape: Shape,
        sparse: Option<usize>,
    ) -> GraphIRResult<AnnotatedNode, B> {
        let sparse = sparse.map(|nnz| NonZeroUsize::try_from(nnz).unwrap());
        let node = self.add_leaf(Some(id.to_string()), shape, false, false, sparse)?;
        self.inputs.insert(node.idx);
        Ok(node)
    }

    pub fn add_weights(&mut self, id: &str, shape: Shape) -> GraphIRResult<AnnotatedNode, B> {
        let node = self.add_leaf(Some(id.to_string()), shape, false, true, None)?;
        self.weights.insert(node.idx);
        Ok(node)
    }

    pub fn add_op(&mut self, operation: impl GraphIROperationCompilable<B>) -> GraphIRResult<AnnotatedNode, B> {
        let op: Rc<dyn GraphIROperationCompilable<B>> = Rc::new(operation);
        let idx = self.inner.add_node(op)?;
        let shape = self.inner.get(idx)?.ty().shape;
        Ok(AnnotatedNode { idx, shape })
    }

    pub fn as_graphviz(&self, prefix: &str) -> Result<String, std::fmt::Error> {
        use std::fmt::Write;

        let mut s = String::new();

        for node in self.inner.topo_order().unwrap() {
            let data = self.get(node).unwrap();
            let node = node.inner();
            let op = data.op();
            let opname = op.shorthand();
            let parents = op.parents();

            if parents.is_empty() {
                writeln!(&mut s, "{prefix}{node} [label=\"{opname}\", style=filled, color=lightblue];")?;
            } else {
                writeln!(&mut s, "{prefix}{node} [label=\"{opname}\"];")?;

                for parent in parents {
                    writeln!(&mut s, "{prefix}{} -> {prefix}{node:?};", parent.inner())?;
                }
            }
        }

        Ok(s)
    }
}

pub type GraphIR<B> = DAGraph<NodeInfo, GraphIROperation<B>>;

pub trait GraphIRMethods<B: BackendMarker> {
    fn create(&mut self, operation: impl GraphIROperationCompilable<B>) -> Result<AnnotatedNode, DAGraphError>;
    fn replace(&mut self, node: NodeId, operation: impl GraphIROperationCompilable<B>) -> Result<(), DAGraphError>;
}

impl<B: BackendMarker> GraphIRMethods<B> for GraphIR<B> {
    fn create(&mut self, operation: impl GraphIROperationCompilable<B>) -> Result<AnnotatedNode, DAGraphError> {
        let idx = self.add_node(GraphIROperation(Rc::new(operation)))?;
        let shape = self.get(idx)?.ty().shape;
        Ok(AnnotatedNode { idx, shape })
    }

    fn replace(&mut self, node: NodeId, operation: impl GraphIROperationCompilable<B>) -> Result<(), DAGraphError> {
        self.replace_op(node, GraphIROperation(Rc::new(operation)))
    }
}

#[derive(Clone)]
pub struct GraphIROperation<B>(Rc<dyn GraphIROperationCompilable<B>>);

impl<B> fmt::Debug for GraphIROperation<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl<B: BackendMarker> From<Rc<dyn GraphIROperationCompilable<B>>> for GraphIROperation<B> {
    fn from(value: Rc<dyn GraphIROperationCompilable<B>>) -> Self {
        Self(value)
    }
}

impl<B: BackendMarker> Operation<NodeInfo> for GraphIROperation<B> {
    fn parents(&self) -> HashSet<NodeId> {
        self.nodes().iter().map(|x| x.idx).collect()
    }

    fn out_type(&self, graph: &DAGraph<NodeInfo, Self>) -> Result<NodeInfo, DAGraphError> {
        let shape = self.output_shape(graph)?;
        let batched = self.output_batched(graph)?;
        let requires_grad = self.output_requires_grad(graph)?;
        let sparse = self.output_layout(graph)?;

        Ok(NodeInfo { requires_grad, sparse, batched, shape })
    }
}

impl<B: BackendMarker> Deref for GraphIROperation<B> {
    type Target = Rc<dyn GraphIROperationCompilable<B>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: BackendMarker> DerefMut for GraphIROperation<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct GraphIRType<B>(PhantomData<B>);
impl<B: BackendMarker> DAGraphType for GraphIRType<B> {
    type Type = NodeInfo;
    type Operation = GraphIROperation<B>;
}

pub type GraphIRResult<T, B> = Result<T, DAGraphManagerError<GraphIRType<B>>>;

#[derive(Debug)]
pub enum GraphIRError {
    Op(GraphIROperationError),
    MultipleRoots,
    CannotBeTopologicallyOrdered,
    NodeAlreadyExists,
    NodeWithIdAlreadyExists(String),
    NodeDataDoesNotMatchExpected,
    NodeDoesNotExist,
    NodeHasInvalidNumberOfChildren,
    AcyclibGraphError(DAGraphError),
}

impl From<DAGraphError> for GraphIRError {
    fn from(value: DAGraphError) -> Self {
        Self::AcyclibGraphError(value)
    }
}

impl From<GraphIROperationError> for GraphIRError {
    fn from(value: GraphIROperationError) -> Self {
        Self::Op(value)
    }
}

impl From<GraphIRError> for DAGraphError {
    fn from(value: GraphIRError) -> Self {
        Self::Message(format!("{value:?}"))
    }
}
