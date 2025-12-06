use crate::{
    common::Shape,
    ir::{
        IrError, IrGraph, IrNodeId, IrType,
        lower::IrLower,
        ops::{IrOperation, broadcast::Broadcast},
    },
    program::{Program, ProgramError, buffer::ProgramBufferId, instruction::ProgramInstruction},
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

impl ProgramInstruction for ReduceDesc {
    fn opname(&self) -> String {
        format!("reduce.{}<{:?}, {:?}>", format!("{:?}", self.op).to_lowercase(), self.start, self.end)
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reduce(IrNodeId, ReduceDesc);

impl Reduce {
    pub fn new(input: IrNodeId, start: impl Into<Shape>, end: impl Into<Shape>, op: ReduceOp) -> Self {
        Self(input, ReduceDesc { start: start.into(), end: end.into(), op })
    }
}

impl IrOperation for Reduce {
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

        if !Broadcast::valid(&self.1.end, &self.1.start) {
            return Err(IrError::Message(format!("{:?} not reducible to {:?}", self.1.start, self.1.end)));
        }

        Ok(vec![IrType::new(self.1.end.size(), node.ty().dtype())])
    }

    fn lower(&self, lower: &mut IrLower, outputs: &[IrNodeId]) -> Result<(), IrError> {
        lower.add_instruction(lower.get_bufs(self.inputs())?, lower.get_bufs(outputs)?, self.1.clone())
    }
}
