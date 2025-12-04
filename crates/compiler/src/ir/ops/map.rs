use crate::{
    common::{MapNode, MapOp},
    ir::{
        IrError, IrGraph,
        lower::IrLower,
        node::{IrNodeId, IrType},
        ops::IrOperation,
    },
    program::{Program, ProgramError, buffer::ProgramBufferId, instruction::ProgramInstruction},
};

impl ProgramInstruction for MapOp<()> {
    fn opname(&self) -> String {
        self.mapname()
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

impl IrOperation for MapOp<MapNode<IrNodeId>> {
    fn opname(&self) -> String {
        self.mapname()
    }

    fn inputs(&self) -> Vec<IrNodeId> {
        self.args()
            .iter()
            .filter_map(|node| match node {
                MapNode::Value(node) => Some(*node),
                MapNode::Constant(_) => None,
            })
            .collect()
    }

    fn output_types(&self, ir: &IrGraph) -> Result<Vec<IrType>, IrError> {
        let dtype = |node| match node {
            MapNode::Value(x) => ir.get_node(x).map(|y| y.ty().dtype()),
            MapNode::Constant(x) => Ok(x.dtype()),
        };

        let size = |node| match node {
            MapNode::Value(x) => ir.get_node(x).map(|y| Some(y.ty().size())),
            MapNode::Constant(_) => Ok(None),
        };

        let (size, dtype) = match *self {
            MapOp::Binary { lhs, rhs, .. } => {
                let size = match (size(lhs)?, size(rhs)?) {
                    (None, None) => Err(IrError::FailedTypeCheck),
                    (Some(x), None) => Ok(x),
                    (None, Some(x)) => Ok(x),
                    (Some(x), Some(y)) => (x == y).then_some(x).ok_or(IrError::InvalidOperationInputs),
                }?;

                let ldtype = dtype(lhs)?;
                let dtype = (ldtype == dtype(rhs)?).then_some(ldtype).ok_or(IrError::FailedTypeCheck)?;

                (size, dtype)
            }
            MapOp::Unary { inp, .. } => (size(inp)?.ok_or(IrError::InvalidOperationInputs)?, dtype(inp)?),
        };

        Ok(vec![IrType::new(size, dtype)])
    }

    fn lower(&self, lower: &mut IrLower, inputs: &[IrNodeId], outputs: &[IrNodeId]) -> Result<(), IrError> {
        lower.add_instruction(lower.get_bufs(inputs)?, lower.get_bufs(outputs)?, self.to(|_| ()))
    }
}
