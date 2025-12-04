use std::collections::HashMap;

use crate::{
    ir::{IrError, IrGraph, node::IrNodeId},
    program::{Program, buffer::ProgramBufferId, instruction::ProgramInstruction},
};

pub struct IrLower<'a> {
    ir: &'a IrGraph,
    program: Program,
    assocs: HashMap<IrNodeId, ProgramBufferId>,
}

impl<'a> IrLower<'a> {
    pub fn new(ir: &'a IrGraph) -> Self {
        let mut program = Program::default();
        let mut assocs = HashMap::default();

        for node in ir.nodes.values() {
            let buf_id = program.add_buffer(node.ty().dtype(), node.ty().size());
            assert!(assocs.insert(node.id(), buf_id).is_none());
        }

        Self { ir, program, assocs }
    }

    pub fn ir(&self) -> &IrGraph {
        self.ir
    }

    pub fn get_buf(&self, node: IrNodeId) -> Result<ProgramBufferId, IrError> {
        self.assocs.get(&node).cloned().ok_or(IrError::NodeDoesNotExist)
    }

    pub fn get_bufs(&self, nodes: impl AsRef<[IrNodeId]>) -> Result<Vec<ProgramBufferId>, IrError> {
        nodes.as_ref().iter().map(|&x| self.get_buf(x)).collect()
    }

    pub fn add_instruction(
        &mut self,
        refs: impl Into<Vec<ProgramBufferId>>,
        muts: impl Into<Vec<ProgramBufferId>>,
        instruction: impl ProgramInstruction + 'static,
    ) -> Result<(), IrError> {
        self.program.add_instruction(refs, muts, instruction).map_err(IrError::Lowering)
    }

    pub fn finalise(self) -> Program {
        self.program
    }
}
