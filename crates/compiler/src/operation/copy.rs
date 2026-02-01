use std::rc::Rc;

use crate::graph::{Op, OpType, TType, TValue};

/// Internal copy operation used for a few special cases. For example:
///
/// If two outputs of an Graph are proven equivalent, but
/// can't eliminate either of them as they must be observable
/// then perform a copy instead of the parent operation twice.
#[derive(Debug, PartialEq)]
pub struct CopyOp(pub TType);

impl OpType for CopyOp {
    fn opname(&self) -> String {
        "copy".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.0]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![self.0]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let size = inputs[0].size();
        assert_eq!(size, outputs[0].size());

        for idx in 0..size {
            outputs[0].write(idx, inputs[0].read(idx));
        }
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = Op::downcast_rc::<Self>(other) { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::{DType, Size};

    use super::*;

    #[test]
    fn evaluate() {
        let a = TValue::F32(vec![1.0; 4]);
        let mut b = TValue::F32(vec![0.0; 4]);

        CopyOp(TType::new(Size::variable(), DType::F32)).evaluate(vec![&a], vec![&mut b]);

        assert_eq!(b, TValue::F32(vec![1.0; 4]));
    }
}
