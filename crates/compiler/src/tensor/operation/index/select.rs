use crate::tensor::{DType, IRTrace, OpType, Size, TNode, TType, TValue, TensorOp};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Select {
    pub dtype: DType,
    pub batch: Size,
    pub inner: Size,
    pub divisor: Size,
}

impl Select {
    pub fn apply<T: Copy + std::fmt::Debug>(&self, input: &[T], indices: &[i32], output: &mut [T]) -> bool {
        let subdim = (self.inner / self.divisor).get();

        assert_eq!(input.len(), (self.batch * self.inner).get());
        assert_eq!(output.len(), (self.batch * subdim).get());
        assert_eq!(indices.len(), self.batch.get());

        for (o, index) in indices.iter().map(|&x| x as usize).enumerate() {
            let oidx = subdim * o;
            let iidx = self.inner.get() * o + subdim * index;
            output[oidx..(subdim + oidx)].copy_from_slice(&input[iidx..(subdim + iidx)]);
        }

        true
    }

    pub fn input_size(&self) -> Size {
        self.batch * self.inner
    }

    pub fn output_size(&self) -> Size {
        self.batch * (self.inner / self.divisor)
    }
}

impl OpType for Select {
    fn opname(&self) -> String {
        let Self { batch, inner, divisor, .. } = *self;
        format!("select<{batch:?}x{inner:?}, {divisor:?}>")
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.input_size(), self.dtype), TType::new(self.batch, DType::I32)]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![TType::new(self.output_size(), self.dtype)]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);

        let TValue::I32(indices) = inputs[1] else { panic!() };

        match self.dtype {
            DType::F32 => {
                let TValue::F32(input) = inputs[0] else { panic!() };
                let TValue::F32(output) = &mut outputs[0] else { panic!() };
                self.apply(input, indices, output);
            }
            DType::I32 => {
                let TValue::I32(input) = inputs[0] else { panic!() };
                let TValue::I32(output) = &mut outputs[0] else { panic!() };
                self.apply(input, indices, output);
            }
        }

        true
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(other) = other.downcast::<Self>() { self == other } else { false }
    }

    fn backward<'a>(&self, inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let Select { dtype, batch, inner, divisor } = *self;
        let op = SelectPad { dtype, batch, inner, divisor };
        output_grads[0].builder().add_op([output_grads[0], inputs[1]], op)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectPad {
    pub dtype: DType,
    pub batch: Size,
    pub inner: Size,
    pub divisor: Size,
}

impl SelectPad {
    pub fn apply<T: Copy + Default + std::fmt::Debug>(&self, input: &[T], indices: &[i32], output: &mut [T]) {
        let subdim = (self.inner / self.divisor).get();
        let inner = self.inner.get();

        assert_eq!(input.len(), (self.batch * subdim).get());
        assert_eq!(output.len(), (self.batch * self.inner).get());
        assert_eq!(indices.len(), self.batch.get());

        for (o, index) in indices.iter().map(|&x| x as usize).enumerate() {
            for i in 0..inner {
                output[inner * o + i] = T::default()
            }

            let iidx = subdim * o;
            let oidx = inner * o + subdim * index;
            output[oidx..(subdim + oidx)].copy_from_slice(&input[iidx..(subdim + iidx)]);
        }
    }

    pub fn input_size(&self) -> Size {
        self.batch * (self.inner / self.divisor)
    }

    pub fn output_size(&self) -> Size {
        self.batch * self.inner
    }
}

impl OpType for SelectPad {
    fn opname(&self) -> String {
        let Self { batch, inner, divisor, .. } = *self;
        format!("selectpad<{batch:?}x{inner:?}, {divisor:?}>")
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.input_size(), self.dtype), TType::new(self.batch, DType::I32)]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![TType::new(self.output_size(), self.dtype)]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);

        let TValue::I32(indices) = inputs[1] else { panic!() };

        match self.dtype {
            DType::F32 => {
                let TValue::F32(input) = inputs[0] else { panic!() };
                let TValue::F32(output) = &mut outputs[0] else { panic!() };
                self.apply(input, indices, output);
            }
            DType::I32 => {
                let TValue::I32(input) = inputs[0] else { panic!() };
                let TValue::I32(output) = &mut outputs[0] else { panic!() };
                self.apply(input, indices, output);
            }
        }

        true
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(other) = other.downcast::<Self>() { self == other } else { false }
    }

    fn backward<'a>(&self, inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let SelectPad { dtype, batch, inner, divisor } = *self;
        let op = Select { dtype, batch, inner, divisor };
        output_grads[0].builder().add_op([output_grads[0], inputs[1]], op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate() {
        let select = Select { dtype: DType::I32, batch: 2.into(), inner: 4.into(), divisor: 2.into() };

        let input = TValue::I32([0, 1, 2, 3, 4, 5, 6, 7].to_vec());
        let indices = TValue::I32([0, 1].into());
        let expected = [0, 1, 6, 7];

        let mut output = TValue::I32(vec![0; expected.len()]);
        select.evaluate(vec![&input, &indices], vec![&mut output]);
        let TValue::I32(output) = output else { panic!() };

        assert_eq!(&output, &expected);
    }

    #[test]
    fn evaluate_pad() {
        let select = SelectPad { dtype: DType::I32, batch: 2.into(), inner: 4.into(), divisor: 2.into() };

        let input = TValue::I32([0, 1, 2, 3].to_vec());
        let indices = TValue::I32([0, 1].into());
        let expected = [0, 1, 0, 0, 0, 0, 2, 3];

        let mut output = TValue::I32(vec![0; expected.len()]);
        select.evaluate(vec![&input, &indices], vec![&mut output]);
        let TValue::I32(output) = output else { panic!() };

        assert_eq!(&output, &expected);
    }

    #[test]
    fn evaluate_inverses() {
        let dtype = DType::I32;
        let batch = Size::from(16);
        let inner = Size::from(16);
        let divisor = Size::from(4);

        let pad = SelectPad { dtype, batch, inner, divisor };
        let select = Select { dtype, batch, inner, divisor };

        let input = TValue::I32((0..64).collect::<Vec<_>>());
        let indices = TValue::I32((0..4).collect::<Vec<_>>().repeat(4));

        let mut output = TValue::I32(vec![0; input.size() * 4]);
        let mut inverse = TValue::I32(vec![0; input.size()]);

        pad.evaluate(vec![&input, &indices], vec![&mut output]);
        select.evaluate(vec![&output, &indices], vec![&mut inverse]);

        assert_eq!(&input, &inverse);
    }
}
