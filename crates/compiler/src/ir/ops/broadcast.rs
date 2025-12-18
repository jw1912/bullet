use crate::{
    common::Shape,
    ir::{IrError, IrGraph, IrNodeId, IrType, ops::IrOperation},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Broadcast(IrNodeId, BroadcastDesc);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BroadcastDesc {
    pub start: Shape,
    pub end: Shape,
}

impl Broadcast {
    pub fn new(input: IrNodeId, start: impl Into<Shape>, end: impl Into<Shape>) -> Self {
        Self(input, BroadcastDesc { start: start.into(), end: end.into() })
    }

    pub fn valid(start: &Shape, end: &Shape) -> bool {
        match (start.inner(), end.inner()) {
            ([x], [y]) => y.is_multiple_of(*x),
            ([x], [y, z]) => x == y || x == z,
            _ => false,
        }
    }
}

impl IrOperation for Broadcast {
    fn opname(&self) -> String {
        format!("broadcast<{:?}, {:?}>", self.1.start, self.1.end)
    }

    fn inputs(&self) -> Vec<IrNodeId> {
        vec![self.0]
    }

    fn output_types(&self, ir: &IrGraph) -> Result<Vec<IrType>, IrError> {
        let node = ir.get_node(self.0)?;

        if node.ty().size() != self.1.start.size() {
            return Err(IrError::FailedTypeCheck);
        }

        if !Broadcast::valid(&self.1.start, &self.1.end) {
            return Err(IrError::Message(format!("{:?} not broadcastable to {:?}", self.1.start, self.1.end)));
        }

        Ok(vec![IrType::new(self.1.end.size(), node.ty().dtype())])
    }
}
