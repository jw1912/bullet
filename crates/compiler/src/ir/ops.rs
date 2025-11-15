mod broadcast;
mod reduce;
mod shape;

use std::{
    fmt::Debug,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

pub use broadcast::Broadcast;
pub use reduce::{Reduce, ReduceOp};
pub use shape::Shape;

use crate::{
    ir::{IrError, IrGraph, IrNodeId, IrType},
    map::MapOp,
};

pub trait IrOperation: Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<IrNodeId>;

    fn output_types(&self, ir: &IrGraph) -> Result<Vec<IrType>, IrError>;
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct IrOpId(usize);

impl Default for IrOpId {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl IrOpId {
    pub fn from_inner(id: usize) -> Self {
        Self(id)
    }

    pub fn inner(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct IrOp {
    id: IrOpId,
    inputs: Vec<IrNodeId>,
    outputs: Vec<IrNodeId>,
    op: Rc<dyn IrOperation>,
}

impl IrOp {
    pub fn new(op: impl IrOperation, ir: &IrGraph) -> Result<Self, IrError> {
        let id = IrOpId::default();
        let op = Rc::new(op);
        let inputs = op.inputs();
        let output_tys = op.output_types(ir)?;
        let outputs = (0..output_tys.len()).map(|_| IrNodeId::default()).collect();

        Ok(Self { id, op, outputs, inputs })
    }

    pub fn id(&self) -> IrOpId {
        self.id
    }

    pub fn inputs(&self) -> &[IrNodeId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[IrNodeId] {
        &self.outputs
    }

    pub fn swap_output_with(&mut self, new: IrNodeId, old: IrNodeId) -> Result<(), IrError> {
        for id in &mut self.outputs {
            if *id == old {
                *id = new;
                return Ok(());
            }
        }

        Err(IrError::NodeDoesNotExist)
    }

    pub fn op(&self) -> &Rc<dyn IrOperation> {
        &self.op
    }
}

#[derive(Debug)]
pub struct Leaf(pub IrType);

impl IrOperation for Leaf {
    fn opname(&self) -> String {
        format!("leaf.{:?}", self.0)
    }

    fn inputs(&self) -> Vec<IrNodeId> {
        Vec::new()
    }

    fn output_types(&self, _ir: &IrGraph) -> Result<Vec<IrType>, IrError> {
        Ok(vec![self.0])
    }
}

impl IrOperation for MapOp<IrNodeId> {
    fn opname(&self) -> String {
        self.opname()
    }

    fn inputs(&self) -> Vec<IrNodeId> {
        self.inputs()
    }

    fn output_types(&self, ir: &IrGraph) -> Result<Vec<IrType>, IrError> {
        let types = self.inputs().iter().map(|input| ir.get_node_type(*input)).collect::<Result<Vec<_>, _>>()?;
        let size = types[0].size();

        if types.iter().any(|x| x.size() != size) {
            return Err(IrError::InvalidOperationInputs);
        }

        let dtype = match *self {
            MapOp::Binary { lhs, rhs, .. } => {
                let dtype = ir.get_node(lhs)?.ty().dtype();
                (dtype == ir.get_node(rhs)?.ty().dtype()).then_some(dtype).ok_or(IrError::FailedTypeCheck)
            }
            MapOp::Unary { inp, .. } => Ok(ir.get_node(inp)?.ty().dtype()),
        }?;

        Ok(vec![IrType::new(size, dtype)])
    }
}
