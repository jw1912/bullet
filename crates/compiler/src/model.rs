mod builder;
pub mod operations;

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    rc::Rc,
};

use crate::{
    ir::{IR, IRError, Node, NodeId, Operation, TypeSystem},
    tensor::{DType, IRTrace, TType, TValue, TensorIR},
};

pub use builder::{Affine, ModelBuilder, ModelNode, Shape};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Sparse(usize),
    Dense(DType),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MType {
    batch: bool,
    rows: usize,
    cols: usize,
    layout: Layout,
}

impl fmt::Display for MType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MType { batch, rows, cols, layout } = *self;
        let batch = if batch { "Bx" } else { "" };
        match layout {
            Layout::Dense(dtype) => write!(f, "{dtype:?}<{batch}{rows}x{cols}>"),
            Layout::Sparse(nnz) => write!(f, "Sparse<{batch}{rows}x{cols}, {nnz}>"),
        }
    }
}

impl fmt::Debug for MType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl MType {
    pub fn new(batch: bool, rows: usize, cols: usize, layout: Layout) -> Self {
        Self { batch, rows, cols, layout }
    }

    pub fn ttype(&self, batch_size: usize) -> TType {
        let MType { batch, rows, cols, layout } = *self;

        let (mut size, dtype) = match layout {
            Layout::Dense(dtype) => (rows * cols, dtype),
            Layout::Sparse(nnz) => (nnz * cols, DType::I32),
        };

        if batch {
            size *= batch_size
        }

        TType::new(size, dtype)
    }

    pub fn single_size(&self) -> usize {
        self.rows * self.cols
    }

    pub fn shape(&self) -> Shape {
        Shape::new(self.rows, self.cols)
    }

    pub fn is_batched(&self) -> bool {
        self.batch
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn is_dense(&self) -> bool {
        matches!(self.layout, Layout::Dense(_))
    }
}

pub trait ModelOperation: 'static + fmt::Debug {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<MType>;

    fn output(&self) -> MType;

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace>;
}

#[derive(Clone, Debug)]
pub struct ModelOp(pub Rc<dyn ModelOperation>);

impl<T: ModelOperation> From<T> for ModelOp {
    fn from(value: T) -> Self {
        Self(Rc::new(value))
    }
}

impl Operation<MType> for ModelOp {
    fn opname(&self) -> String {
        self.0.opname()
    }

    fn inputs(&self) -> Vec<MType> {
        self.0.inputs()
    }

    fn outputs(&self) -> Vec<MType> {
        vec![self.0.output()]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Model;
impl TypeSystem for Model {
    type Type = MType;
    type OpData = ModelOp;
}

#[derive(Clone, Debug)]
pub enum InitSettings {
    Zeroed,
    Normal { mean: f32, stdev: f32 },
    Uniform { mean: f32, stdev: f32 },
    Custom(TValue),
}

#[derive(Clone, Debug, Default)]
pub struct ModelIR {
    ir: IR<Model>,
    weights: BTreeMap<NodeId, (String, InitSettings)>,
    inputs: BTreeMap<NodeId, String>,
    requires_grad: BTreeSet<NodeId>,
    stop_grad: bool,
}

impl ModelIR {
    pub fn inner(&self) -> &IR<Model> {
        &self.ir
    }

    pub fn node(&self, node: NodeId) -> &Node<Model> {
        self.ir.node(node).unwrap()
    }

    pub fn weights(&self) -> &BTreeMap<NodeId, (String, InitSettings)> {
        &self.weights
    }

    pub fn inputs(&self) -> &BTreeMap<NodeId, String> {
        &self.inputs
    }

    pub fn add_weight(&mut self, name: impl Into<String>, rows: usize, cols: usize, init: InitSettings) -> NodeId {
        let ty = MType { batch: false, rows, cols, layout: Layout::Dense(DType::F32) };
        let node = self.ir.add_op([], operations::Input(ty).into()).unwrap()[0];
        self.weights.insert(node, (name.into(), init));

        if !self.stop_grad {
            self.requires_grad.insert(node);
        }

        node
    }

    pub fn add_input(
        &mut self,
        name: impl Into<String>,
        batch: bool,
        rows: usize,
        cols: usize,
        layout: Layout,
    ) -> NodeId {
        let ty = MType { batch, rows, cols, layout };
        let node = self.ir.add_op([], operations::Input(ty).into()).unwrap()[0];
        self.inputs.insert(node, name.into());
        node
    }

    pub fn add_op(&mut self, inputs: impl AsRef<[NodeId]>, op: impl ModelOperation) -> Result<NodeId, IRError> {
        let req_grad = inputs.as_ref().iter().any(|i| self.requires_grad.contains(i));
        let node = self.ir.add_op(inputs, op.into()).map(|x| x[0])?;

        if !self.stop_grad && req_grad {
            self.requires_grad.insert(node);
        }

        Ok(node)
    }

    pub fn lower(&self, batch_size: usize) -> Result<(TensorIR, BTreeMap<NodeId, NodeId>), IRTrace> {
        let mut ir = TensorIR::default();

        let mut map = BTreeMap::default();

        for op in self.ir.ordered_operations()? {
            let tinputs = op.inputs().iter().map(|node| *map.get(node).unwrap()).collect::<Vec<_>>();
            let toutput = op.data().0.lower(batch_size, &mut ir, tinputs)?;
            map.insert(op.outputs()[0], toutput);
        }

        Ok((ir, map))
    }
}
