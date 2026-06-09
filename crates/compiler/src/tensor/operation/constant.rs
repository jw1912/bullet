use crate::tensor::{DValue, IRTrace, OpType, Size, TNode, TType, TValue, TensorOp};

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

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
        assert_eq!(inputs.len(), 0);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].size(), self.0.size());
        assert_eq!(outputs[0].dtype(), self.0.dtype());

        *outputs[0] = self.0.clone();
        true
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(other) = other.downcast::<Self>() { self == other } else { false }
    }

    fn backward<'a>(&self, _inputs: Vec<TNode<'a>>, _output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        Ok(Vec::new())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarConstant(pub DValue, pub Size);

impl ScalarConstant {
    pub fn ty(&self) -> TType {
        TType::new(self.1, self.0.dtype())
    }

    pub fn to_tensor(&self) -> TValue {
        match self.0 {
            DValue::F32(x) => TValue::F32(vec![x; self.1.get()]),
            DValue::I32(x) => TValue::I32(vec![x; self.1.get()]),
        }
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

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
        assert_eq!(inputs.len(), 0);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].dtype(), self.0.dtype());

        for i in 0..outputs[0].size() {
            outputs[0].write(i, self.0);
        }

        true
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(other) = other.downcast::<Self>() { self == other } else { false }
    }

    fn backward<'a>(&self, _inputs: Vec<TNode<'a>>, _output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        Ok(Vec::new())
    }
}
