use std::rc::Rc;

use crate::{
    core::{DTypeTensor, DTypeValue, Size},
    ir::graph::{IrOperationType, IrType},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Constant(pub DTypeTensor);

impl Constant {
    pub fn ty(&self) -> IrType {
        IrType::new(self.0.size(), self.0.dtype())
    }
}

impl IrOperationType for Constant {
    fn opname(&self) -> String {
        format!("constant<{:?}>", self.ty())
    }

    fn inputs(&self) -> Vec<IrType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![self.ty()]
    }

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]) {
        assert_eq!(inputs.len(), 0);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].size(), self.0.size());
        assert_eq!(outputs[0].dtype(), self.0.dtype());

        *outputs[0] = self.0.clone();
    }

    fn equals(&self, _: &Rc<dyn IrOperationType>) -> bool {
        false
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarConstant(pub DTypeValue, pub Size);

impl ScalarConstant {
    pub fn ty(&self) -> IrType {
        IrType::new(self.1, self.0.dtype())
    }

    pub fn to_tensor(&self) -> Option<DTypeTensor> {
        self.1.evaluate_constant().map(|size| match self.0 {
            DTypeValue::F32(x) => DTypeTensor::F32(vec![x; size]),
            DTypeValue::I32(x) => DTypeTensor::I32(vec![x; size]),
        })
    }
}

impl IrOperationType for ScalarConstant {
    fn opname(&self) -> String {
        format!("constant<[{:?}; {:?}]>", self.0, self.1)
    }

    fn inputs(&self) -> Vec<IrType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![self.ty()]
    }

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]) {
        assert_eq!(inputs.len(), 0);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].dtype(), self.0.dtype());

        for i in 0..outputs[0].size() {
            outputs[0].write(i, self.0);
        }
    }

    fn equals(&self, _: &Rc<dyn IrOperationType>) -> bool {
        false
    }
}
