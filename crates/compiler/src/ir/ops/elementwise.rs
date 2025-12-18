use crate::{
    Size,
    elementwise::{Binary, ElementwiseBuilder, ElementwiseDescription, ElementwiseId, ElementwiseNode, Unary},
    ir::{
        IrError, IrGraph,
        node::{IrNode, IrNodeId, IrType},
        ops::IrOperation,
    },
};

#[derive(Debug)]
pub struct IrElementwise {
    size: Size,
    inputs: Vec<(ElementwiseId, IrNodeId)>,
    op: ElementwiseDescription,
    outputs: Vec<ElementwiseId>,
}

impl IrElementwise {
    pub fn new<const M: usize, const N: usize, F>(inputs: [&IrNode; M], f: F) -> Result<Self, IrError>
    where
        for<'a> F: Fn([ElementwiseNode<'a>; M]) -> Option<[ElementwiseNode<'a>; N]>,
    {
        let builder = ElementwiseBuilder::default();

        let inps = inputs.map(|x| (builder.add_input(x.ty().dtype()), x.id()));
        let outs = f(inps.map(|x| x.0)).ok_or(IrError::FailedTypeCheck)?;

        let sizes = inputs.map(|x| x.ty().size());
        let size = sizes[0];
        for other in sizes.into_iter().skip(1) {
            if size != other {
                return Err(IrError::InvalidOperationInputs);
            }
        }

        let inputs = inps.map(|(x, y)| (x.node, y)).into();
        let outputs = outs.map(|x| x.node).into();

        Ok(Self { size, inputs, op: builder.build(), outputs })
    }

    pub fn unary(input: &IrNode, op: Unary) -> Result<Self, IrError> {
        Self::new([input], |[input]| input.unary(op).map(|x| [x]))
    }

    pub fn binary(lhs: &IrNode, rhs: &IrNode, op: Binary) -> Result<Self, IrError> {
        Self::new([lhs, rhs], |[lhs, rhs]| lhs.binary(rhs, op).map(|x| [x]))
    }
}

impl IrOperation for IrElementwise {
    fn opname(&self) -> String {
        "elementwise".into()
    }

    fn inputs(&self) -> Vec<IrNodeId> {
        self.inputs.iter().map(|x| x.1).collect()
    }

    fn output_types(&self, _ir: &IrGraph) -> Result<Vec<IrType>, IrError> {
        Ok(self.outputs.iter().map(|&out| IrType::new(self.size, self.op.get_dtype(out.into()))).collect())
    }
}
