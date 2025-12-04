pub mod buffer;
pub mod instruction;

use std::{collections::HashMap, fmt};

use crate::common::{DType, Size};

use buffer::{ProgramBuffer, ProgramBufferId};
use instruction::{ProgramInst, ProgramInstruction};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramError {
    Aliasing,
    BufferDoesNotExist,
    InvalidBuffers,
}

#[derive(Default)]
pub struct Program {
    allocations: HashMap<ProgramBufferId, ProgramBuffer>,
    instructions: Vec<ProgramInst>,
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "program buffers")?;

        for buf in self.allocations.values() {
            writeln!(f, "{:?} = {:?}[{:?}]", buf.id(), buf.dtype(), buf.len())?;
        }

        writeln!(f, "program start")?;

        for inst in &self.instructions {
            writeln!(f, "{inst:?}")?;
        }

        write!(f, "program end")
    }
}

impl Program {
    pub fn validate(&self) -> Result<(), ProgramError> {
        for instruction in &self.instructions {
            instruction.validate(self)?;
        }

        Ok(())
    }

    pub fn get_buffer(&self, id: ProgramBufferId) -> Result<&ProgramBuffer, ProgramError> {
        self.allocations.get(&id).ok_or(ProgramError::BufferDoesNotExist)
    }

    pub fn add_buffer(&mut self, dtype: DType, len: Size) -> ProgramBufferId {
        let buf = ProgramBuffer::new(dtype, len);
        assert!(self.allocations.insert(buf.id(), buf).is_none(), "Unique ID!");
        buf.id()
    }

    pub fn add_instruction(
        &mut self,
        refs: impl Into<Vec<ProgramBufferId>>,
        muts: impl Into<Vec<ProgramBufferId>>,
        instruction: impl ProgramInstruction + 'static,
    ) -> Result<(), ProgramError> {
        self.instructions.push(ProgramInst::new(refs, muts, instruction));
        self.instructions.last().unwrap().validate(self)
    }
}
