use crate::ir::{
    IrError, IrGraph, IrNodeId, IrType,
    ops::{IrOperation, shape::Shape},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Broadcast {
    input: IrNodeId,
    start: Shape,
    end: Shape,
}

impl Broadcast {
    pub fn new(input: IrNodeId, start: impl Into<Shape>, end: impl Into<Shape>) -> Self {
        Self { input, start: start.into(), end: end.into() }
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
        format!("broadcast[{:?}, {:?}]", self.start, self.end)
    }

    fn inputs(&self) -> Vec<IrNodeId> {
        vec![self.input]
    }

    fn output_types(&self, ir: &IrGraph) -> Result<Vec<IrType>, IrError> {
        let node = ir.get_node(self.input)?;

        if node.ty().size() != self.start.size() {
            return Err(IrError::FailedTypeCheck);
        }

        if !Broadcast::valid(&self.start, &self.end) {
            return Err(IrError::Message(format!("{:?} not broadcastable to {:?}", self.start, self.end)));
        }

        Ok(vec![IrType::new(self.end.size(), node.ty().dtype())])
    }
}
