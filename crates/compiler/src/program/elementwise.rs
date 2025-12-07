use std::collections::HashMap;

use crate::{
    elementwise::{ElementwiseId, ElementwiseKernel, Operation, description::Input},
    program::{Program, ProgramError, buffer::ProgramBufferId, instruction::ProgramInstruction},
};

impl ProgramInstruction for ElementwiseKernel {
    fn opname(&self) -> String {
        use std::fmt::Write;

        let mut s = String::new();

        writeln!(&mut s, "elementwise[").unwrap();

        let mut ids = HashMap::<ElementwiseId, String>::default();

        let mut get = |x| match x {
            Input::Index(x) => {
                if let Some(y) = ids.get(&x) {
                    y.to_string()
                } else {
                    let y = ids.len().to_string();
                    ids.insert(x, y.clone());
                    y
                }
            }
            Input::Constant(x) => format!("{x}"),
        };

        let reads = self.reads().collect::<HashMap<_, _>>();

        self.desc().traverse(|x, op| {
            write!(&mut s, "  ").unwrap();
            let y = get(Input::Index(x));
            match op {
                Operation::Leaf(_) => {
                    let (mutable, idx) = *reads.get(&x).cloned().unwrap();
                    writeln!(&mut s, ".{y} = {}[{idx}]", if mutable { "muts" } else { "refs" }).unwrap();
                }
                Operation::Unary { input, op } => {
                    let i = get(input);
                    writeln!(&mut s, ".{y} = {op:?}(.{i})").unwrap()
                }
                Operation::Binary { lhs, rhs, op } => {
                    let l = get(lhs);
                    let r = get(rhs);
                    writeln!(&mut s, ".{y} = {op:?}(.{l}, .{r})").unwrap()
                }
            }
        });

        for (&id, &idx) in self.writes() {
            writeln!(&mut s, "  muts[{idx}] = .{}", get(Input::Index(id))).unwrap();
        }

        write!(&mut s, "]").unwrap();

        s
    }

    fn validate(
        &self,
        program: &Program,
        refs: &[ProgramBufferId],
        muts: &[ProgramBufferId],
    ) -> Result<(), ProgramError> {
        if refs.len() != self.num_refs() || muts.len() != self.num_muts() {
            println!("FUCK1: {} != {} or {} != {}", refs.len(), self.num_refs(), muts.len(), self.num_muts());
            return Err(ProgramError::InvalidBuffers);
        }

        for (&read, &(mutable, idx)) in self.reads() {
            let buf = program.get_buffer(if mutable { muts[idx] } else { refs[idx] })?;

            if self.desc().get_dtype(read.into()) != buf.dtype() || self.size() != buf.len() {
                println!("FUCK2");
                return Err(ProgramError::InvalidBuffers);
            }
        }

        for (&write, &idx) in self.writes() {
            let buf = program.get_buffer(muts[idx])?;

            if self.desc().get_dtype(write.into()) != buf.dtype() || self.size() != buf.len() {
                println!("FUCK3");
                return Err(ProgramError::InvalidBuffers);
            }
        }

        Ok(())
    }
}
