use std::{collections::HashSet, fmt};

use crate::program::{Program, ProgramBufferId, ProgramError};

pub trait ProgramInstruction {
    fn opname(&self) -> String;

    fn validate(
        &self,
        program: &Program,
        refs: &[ProgramBufferId],
        muts: &[ProgramBufferId],
    ) -> Result<(), ProgramError>;
}

pub struct ProgramInst {
    refs: Vec<ProgramBufferId>,
    muts: Vec<ProgramBufferId>,
    inst: Box<dyn ProgramInstruction>,
}

impl fmt::Debug for ProgramInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inst.opname())?;

        write!(f, "(")?;

        for bufs in [&self.refs, &self.muts] {
            write!(f, "[")?;

            for (i, buf) in bufs.iter().enumerate() {
                write!(f, "{buf:?}")?;
                if i != bufs.len() - 1 {
                    write!(f, ", ")?;
                }
            }

            write!(f, "]")?;
        }

        write!(f, ")")
    }
}

impl ProgramInst {
    pub fn new(
        refs: impl Into<Vec<ProgramBufferId>>,
        muts: impl Into<Vec<ProgramBufferId>>,
        instruction: impl ProgramInstruction + 'static,
    ) -> Self {
        Self { refs: refs.into(), muts: muts.into(), inst: Box::new(instruction) }
    }

    pub fn validate(&self, program: &Program) -> Result<(), ProgramError> {
        for &this_ref in &self.refs {
            let _ = program.get_buffer(this_ref)?;
        }

        for &this_mut in &self.muts {
            let _ = program.get_buffer(this_mut)?;
        }

        let refs: HashSet<_> = self.refs.iter().collect();
        let muts: HashSet<_> = self.muts.iter().collect();

        if muts.len() != self.muts.len() || refs.intersection(&muts).next().is_some() {
            return Err(ProgramError::Aliasing);
        }

        self.inst.validate(program, &self.refs, &self.muts)
    }

    pub fn refs(&self) -> &[ProgramBufferId] {
        &self.refs
    }

    pub fn muts(&self) -> &[ProgramBufferId] {
        &self.muts
    }

    pub fn inst(&self) -> &dyn ProgramInstruction {
        self.inst.as_ref()
    }
}
