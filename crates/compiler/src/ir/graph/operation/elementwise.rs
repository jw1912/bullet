use std::{collections::HashSet, rc::Rc};

use crate::{
    common::{DTypeTensor, Size},
    elementwise::{ElementwiseBuilder, ElementwiseDescription, ElementwiseId, ElementwiseNode},
    ir::graph::{IrError, IrOperation, IrOperationType, IrType},
};

#[derive(Clone, Debug, PartialEq)]
pub struct IrElementwise {
    size: Size,
    inputs: Vec<ElementwiseId>,
    op: ElementwiseDescription,
    outputs: Vec<ElementwiseId>,
}

impl IrElementwise {
    pub(in crate::ir) fn size(&self) -> Size {
        self.size
    }

    pub(in crate::ir) fn input_ids(&self) -> &[ElementwiseId] {
        &self.inputs
    }

    pub(in crate::ir) fn output_ids(&self) -> &[ElementwiseId] {
        &self.outputs
    }

    pub(in crate::ir) fn desc(&self) -> &ElementwiseDescription {
        &self.op
    }

    pub fn new<const M: usize, const N: usize, F>(inputs: [IrType; M], f: F) -> Result<Self, IrError>
    where
        for<'a> F: Fn([ElementwiseNode<'a>; M]) -> Option<[ElementwiseNode<'a>; N]>,
    {
        let builder = ElementwiseBuilder::default();

        let inps = inputs.map(|x| builder.add_input(x.dtype()));
        let outs = f(inps).ok_or("IrElementwise::new: failed dtype check!")?;

        let sizes = inputs.map(|x| x.size());
        let size = sizes[0];
        for other in sizes.into_iter().skip(1) {
            if size != other {
                return Err("IrElementwise::new: failed size check!".into());
            }
        }

        let inputs = inps.map(|x| x.node).into();
        let outputs = outs.map(|x| x.node).into();

        Ok(Self { size, inputs, op: builder.build(), outputs })
    }
}

impl IrOperationType for IrElementwise {
    fn opname(&self) -> String {
        "elementwise".into()
    }

    fn inputs(&self) -> Vec<IrType> {
        self.inputs.iter().map(|&x| IrType::new(self.size, self.op.get_dtype(x))).collect()
    }

    fn outputs(&self) -> Vec<IrType> {
        self.outputs.iter().map(|&out| IrType::new(self.size, self.op.get_dtype(out))).collect()
    }

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]) {
        let input_size = inputs.iter().map(|x| x.size()).collect::<HashSet<_>>();
        let output_size = outputs.iter().map(|x| x.size()).collect::<HashSet<_>>();

        let sizes = input_size.union(&output_size).collect::<Vec<_>>();
        assert_eq!(sizes.len(), 1);
        let size = *sizes[0];

        for idx in 0..size {
            let inputs = self.inputs.iter().zip(inputs).map(|(id, input)| (*id, input.read(idx))).collect();

            let values = self.op.evaluate(inputs, &self.outputs).unwrap();

            for (out, val) in outputs.iter_mut().zip(values) {
                out.write(idx, val);
            }
        }
    }

    fn equals(&self, other: &Rc<dyn IrOperationType>) -> bool {
        if let Some(other) = IrOperation::downcast::<Self>(other) { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use crate::common::DType;

    use super::*;

    #[test]
    fn basic() {
        let ty = IrType::new(Size::variable(), DType::F32);

        let elmt = IrElementwise::new([ty, ty, ty], |[a, b, c]| Some([a * b + c])).unwrap();

        let a = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);
        let b = DTypeTensor::F32(vec![2.0, 0.5, 4.0, 1.0]);
        let c = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);

        let mut output = DTypeTensor::F32(vec![0.0; 4]);

        elmt.evaluate(&[&a, &b, &c], &mut [&mut output]);

        assert_eq!(output, DTypeTensor::F32(vec![3.0, 3.0, 15.0, 8.0]));
    }
}
