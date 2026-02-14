use std::rc::Rc;

use crate::graph::{DValue, Op, OpType, Size, TType, TValue};

#[derive(Clone, Debug, PartialEq)]
pub struct Constant(pub TValue);

impl Constant {
    pub fn ty(&self) -> TType {
        TType::new(self.0.size(), self.0.dtype())
    }
}

impl OpType for Constant {
    fn opname(&self) -> String {
        format!("constant<{:?}>", self.ty())
    }

    fn inputs(&self) -> Vec<TType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<TType> {
        vec![self.ty()]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        assert_eq!(inputs.len(), 0);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].size(), self.0.size());
        assert_eq!(outputs[0].dtype(), self.0.dtype());

        *outputs[0] = self.0.clone();
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = Op::downcast_rc::<Self>(other) { self == other } else { false }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarConstant(pub DValue, pub Size);

impl ScalarConstant {
    pub fn ty(&self) -> TType {
        TType::new(self.1, self.0.dtype())
    }

    pub fn to_tensor(&self) -> Option<TValue> {
        self.1.evaluate_constant().map(|size| match self.0 {
            DValue::F32(x) => TValue::F32(vec![x; size]),
            DValue::I32(x) => TValue::I32(vec![x; size]),
        })
    }
}

impl OpType for ScalarConstant {
    fn opname(&self) -> String {
        format!("constant<[{:?}; {:?}]>", self.0, self.1)
    }

    fn inputs(&self) -> Vec<TType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<TType> {
        vec![self.ty()]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        assert_eq!(inputs.len(), 0);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].dtype(), self.0.dtype());

        for i in 0..outputs[0].size() {
            outputs[0].write(i, self.0);
        }
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = Op::downcast_rc::<Self>(other) { self == other } else { false }
    }
}
