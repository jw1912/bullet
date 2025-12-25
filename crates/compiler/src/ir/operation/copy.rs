use std::rc::Rc;

use crate::{
    common::DTypeTensor,
    ir::{
        node::IrType,
        operation::{IrOperation, IrOperationType},
    },
};

/// Internal copy operation used for a few special cases. For example:
///
/// If two outputs of an IrGraph are proven equivalent,
/// can't eliminate either of them as they must be observable
/// so perform a copy instead of the parent operation twice.
#[derive(Debug, PartialEq)]
pub struct IrCopy(pub IrType);

impl IrOperationType for IrCopy {
    fn opname(&self) -> String {
        "copy".to_string()
    }

    fn inputs(&self) -> Vec<IrType> {
        vec![self.0]
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![self.0]
    }

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let size = inputs[0].size();
        assert_eq!(size, outputs[0].size());

        for idx in 0..size {
            outputs[0].write(idx, inputs[0].read(idx));
        }
    }

    fn equals(&self, other: &Rc<dyn IrOperationType>) -> bool {
        if let Some(other) = IrOperation::downcast::<Self>(other) { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use crate::common::{DType, Size};

    use super::*;

    #[test]
    fn evaluate() {
        let a = DTypeTensor::F32(vec![1.0; 4]);
        let mut b = DTypeTensor::F32(vec![0.0; 4]);

        IrCopy(IrType::new(Size::variable(), DType::F32)).evaluate(&[&a], &mut [&mut b]);

        assert_eq!(b, DTypeTensor::F32(vec![1.0; 4]));
    }
}
