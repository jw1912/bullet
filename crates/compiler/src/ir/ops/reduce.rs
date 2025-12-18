use crate::{
    common::Shape,
    ir::{
        IrError, IrGraph, IrNodeId, IrType,
        ops::{IrOperation, broadcast::Broadcast},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReduceDesc {
    pub start: Shape,
    pub end: Shape,
    pub op: ReduceOp,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reduce(IrNodeId, ReduceDesc);

impl Reduce {
    pub fn new(input: IrNodeId, start: impl Into<Shape>, end: impl Into<Shape>, op: ReduceOp) -> Self {
        Self(input, ReduceDesc { start: start.into(), end: end.into(), op })
    }
}

impl IrOperation for Reduce {
    fn opname(&self) -> String {
        format!("reduce.{:?}<{:?}, {:?}>", self.1.op, self.1.start, self.1.end)
    }

    fn inputs(&self) -> Vec<IrNodeId> {
        vec![self.0]
    }

    fn output_types(&self, ir: &IrGraph) -> Result<Vec<IrType>, IrError> {
        let node = ir.get_node(self.0)?;

        if node.ty().size() != self.1.start.size() {
            return Err(IrError::FailedTypeCheck);
        }

        if !Broadcast::valid(&self.1.end, &self.1.start) {
            return Err(IrError::Message(format!("{:?} not reducible to {:?}", self.1.start, self.1.end)));
        }

        Ok(vec![IrType::new(self.1.end.size(), node.ty().dtype())])
    }
}
