mod builder;
mod lower;
mod operations;

use std::{collections::BTreeSet, fmt, rc::Rc};

use crate::{
    ir::{IR, IRError, NodeId, Operation, TypeSystem},
    model::operations::Input,
    tensor::{DType, IRTrace, TType, TensorIR},
};

pub use builder::{ModelNode, ModelBuilder, Shape};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Sparse(usize),
    Dense(DType),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
            Layout::Dense(dtype) => write!(f, "{dtype:?}[{batch}{rows}x{cols}]"),
            Layout::Sparse(nnz) => write!(f, "sparse<{batch}{rows}x{cols}, {nnz}>"),
        }
    }
}

impl MType {
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

    pub fn is_dense(&self) -> bool {
        if let Layout::Dense(_) = self.layout { true } else { false }
    }
}

pub trait ModelOperation: 'static + fmt::Debug {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<MType>;

    fn output(&self) -> MType;

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace>;

    fn gradient(
        &self,
        ir: &mut ModelIR,
        inputs: Vec<NodeId>,
        output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace>;
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

#[derive(Clone, Debug, Default)]
pub struct ModelIR {
    ir: IR<Model>,
    weights: BTreeSet<NodeId>,
    inputs: BTreeSet<NodeId>,
    outputs: BTreeSet<NodeId>,
    requires_grad: BTreeSet<NodeId>,
    pub stop_grad: bool,
}

impl ModelIR {
    pub fn add_weight(&mut self, rows: usize, cols: usize) -> NodeId {
        let ty = MType { batch: false, rows, cols, layout: Layout::Dense(DType::F32) };
        let node = self.ir.add_op([], Input(ty).into()).unwrap()[0];
        self.weights.insert(node);

        if !self.stop_grad {
            self.requires_grad.insert(node);
        }

        node
    }

    pub fn add_input(&mut self, batch: bool, rows: usize, cols: usize, layout: Layout) -> NodeId {
        let ty = MType { batch, rows, cols, layout };
        let node = self.ir.add_op([], Input(ty).into()).unwrap()[0];
        self.inputs.insert(node);
        node
    }

    pub fn register_output(&mut self, node: NodeId) -> Result<(), IRError> {
        if self.weights.contains(&node) || self.inputs.contains(&node) {
            return Err("Cannot register a weight/input as an output!".into());
        }

        self.ir.node(node)?;
        self.outputs.insert(node);

        Ok(())
    }

    pub fn add_op(&mut self, inputs: impl AsRef<[NodeId]>, op: impl ModelOperation) -> Result<NodeId, IRError> {
        let req_grad = inputs.as_ref().iter().any(|i| self.requires_grad.contains(i));
        let node = self.ir.add_op(inputs, op.into()).map(|x| x[0])?;

        if !self.stop_grad && req_grad {
            self.requires_grad.insert(node);
        }

        Ok(node)
    }
}
