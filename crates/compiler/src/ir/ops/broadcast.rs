use crate::{
    common::Shape,
    ir::{IrError, IrGraph, IrNodeId, IrType, lower::IrLower, ops::IrOperation},
    program::{Program, ProgramError, buffer::ProgramBufferId, instruction::ProgramInstruction},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Broadcast(IrNodeId, BroadcastDesc);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BroadcastDesc {
    pub start: Shape,
    pub end: Shape,
}

impl ProgramInstruction for BroadcastDesc {
    fn opname(&self) -> String {
        format!("broadcast<{:?}, {:?}>", self.start, self.end)
    }

    fn validate(
        &self,
        program: &Program,
        refs: &[ProgramBufferId],
        muts: &[ProgramBufferId],
    ) -> Result<(), ProgramError> {
        if refs.len() != 1 || muts.len() != 1 {
            return Err(ProgramError::InvalidBuffers);
        }

        if !(self.start.size().is_le(program.get_buffer(refs[0])?.len())
            && self.end.size().is_le(program.get_buffer(muts[0])?.len()))
        {
            return Err(ProgramError::InvalidBuffers);
        }

        Ok(())
    }
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
        self.1.opname()
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

    fn lower(&self, lower: &mut IrLower, inputs: &[IrNodeId], outputs: &[IrNodeId]) -> Result<(), IrError> {
        if inputs[0] != self.0 {
            return Err(IrError::Lowering(ProgramError::InvalidBuffers));
        }

        lower.add_instruction(lower.get_bufs(inputs)?, lower.get_bufs(outputs)?, self.1.clone())
    }
}
