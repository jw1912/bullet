use std::fmt::Debug;

use crate::{
    ir::{
        IrError, IrGraph,
        lower::IrLower,
        node::{IrNodeId, IrType},
        ops::IrOperation,
    },
    program::{Program, ProgramError, buffer::ProgramBufferId, instruction::ProgramInstruction},
};

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapOp<T> {
    Unary { inp: T, op: UnaryOp },
    Binary { lhs: T, rhs: T, op: BinaryOp },
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Log,
    Sgn,
    Abs,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Min,
    Max,
    AbsPow,
}

impl<T: Copy> MapOp<T> {
    pub fn opname(&self) -> String {
        format!(
            "map.{}",
            match *self {
                MapOp::Unary { op, .. } => format!("{op:?}"),
                MapOp::Binary { op, .. } => format!("{op:?}"),
            }
            .to_lowercase()
        )
    }

    pub fn inputs(&self) -> Vec<T> {
        match *self {
            MapOp::Binary { lhs, rhs, .. } => vec![lhs, rhs],
            MapOp::Unary { inp, .. } => vec![inp],
        }
    }

    pub fn to<U>(&self, f: impl Fn(&T) -> U) -> MapOp<U> {
        match self {
            Self::Unary { inp, op } => MapOp::Unary { inp: f(inp), op: *op },
            Self::Binary { lhs, rhs, op } => MapOp::Binary { lhs: f(lhs), rhs: f(rhs), op: *op },
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Self::Unary { .. } => 1,
            Self::Binary { .. } => 2,
        }
    }
}

impl ProgramInstruction for MapOp<()> {
    fn opname(&self) -> String {
        self.opname()
    }

    fn validate(
        &self,
        _program: &Program,
        refs: &[ProgramBufferId],
        muts: &[ProgramBufferId],
    ) -> Result<(), ProgramError> {
        if refs.len() != self.arity() || muts.len() != 1 {
            return Err(ProgramError::InvalidBuffers);
        }

        Ok(())
    }
}

impl IrOperation for MapOp<IrNodeId> {
    fn opname(&self) -> String {
        self.opname()
    }

    fn inputs(&self) -> Vec<IrNodeId> {
        self.inputs()
    }

    fn output_types(&self, ir: &IrGraph) -> Result<Vec<IrType>, IrError> {
        let types = self.inputs().iter().map(|input| ir.get_node_type(*input)).collect::<Result<Vec<_>, _>>()?;
        let size = types[0].size();

        if types.iter().any(|x| x.size() != size) {
            return Err(IrError::InvalidOperationInputs);
        }

        let dtype = match *self {
            MapOp::Binary { lhs, rhs, .. } => {
                let dtype = ir.get_node(lhs)?.ty().dtype();
                (dtype == ir.get_node(rhs)?.ty().dtype()).then_some(dtype).ok_or(IrError::FailedTypeCheck)
            }
            MapOp::Unary { inp, .. } => Ok(ir.get_node(inp)?.ty().dtype()),
        }?;

        Ok(vec![IrType::new(size, dtype)])
    }

    fn lower(&self, lower: &mut IrLower, inputs: &[IrNodeId], outputs: &[IrNodeId]) -> Result<(), IrError> {
        lower.add_instruction(lower.get_bufs(inputs)?, lower.get_bufs(outputs)?, self.to(|_| ()))
    }
}
