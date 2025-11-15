use crate::ir::{
    IrError, IrGraph, IrNodeId, IrType,
    ops::{IrOperation, broadcast::Broadcast, shape::Shape},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reduce {
    input: IrNodeId,
    start: Shape,
    end: Shape,
    op: ReduceOp,
}

impl Reduce {
    pub fn new(input: IrNodeId, start: impl Into<Shape>, end: impl Into<Shape>, op: ReduceOp) -> Self {
        Self { input, start: start.into(), end: end.into(), op }
    }
}

impl IrOperation for Reduce {
    fn opname(&self) -> String {
        format!("reduce.{}[{:?}, {:?}]", format!("{:?}", self.op).to_lowercase(), self.start, self.end)
    }

    fn inputs(&self) -> Vec<IrNodeId> {
        vec![self.input]
    }

    fn output_types(&self, ir: &IrGraph) -> Result<Vec<IrType>, IrError> {
        let node = ir.get_node(self.input)?;

        if node.ty().size() != self.start.size() {
            return Err(IrError::FailedTypeCheck);
        }

        if !Broadcast::valid(&self.end, &self.start) {
            return Err(IrError::Message(format!("{:?} not reducible to {:?}", self.start, self.end)));
        }

        Ok(vec![IrType::new(self.end.size(), node.ty().dtype())])
    }
}
