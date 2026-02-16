use std::rc::Rc;

use crate::tensor::{DType, OpType, Size, TType, TValue, TensorOp};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Select {
    pub dtype: DType,
    pub batch: Size,
    pub inner: Size,
    pub divisor: Size,
}

impl Select {
    pub fn apply<T: Copy + std::fmt::Debug>(&self, input: &[T], indices: &[i32], output: &mut [T]) {
        let subdim = self.inner / self.divisor;
        let input_size = self.batch * self.inner;
        let output_size = self.batch * subdim;

        let var = match (input_size.get_var_size(input.len()), output_size.get_var_size(output.len())) {
            (None, None) => 1,
            (Some(x), Some(y)) => {
                assert_eq!(x, y);
                x
            }
            (Some(x), None) => x,
            (None, Some(x)) => x,
        };

        assert_eq!(input_size.evaluate(var), input.len());
        assert_eq!(output_size.evaluate(var), output.len());

        let batch = self.batch.evaluate(var);
        let inner = self.inner.evaluate(var);
        let subdim = subdim.evaluate(var);

        assert_eq!(input.len(), batch * inner);
        assert_eq!(output.len(), batch * subdim);
        assert_eq!(indices.len(), batch);

        for (o, index) in indices.iter().map(|&x| x as usize).enumerate() {
            let oidx = subdim * o;
            let iidx = inner * o + subdim * index;
            output[oidx..(subdim + oidx)].copy_from_slice(&input[iidx..(subdim + iidx)]);
        }
    }
}

impl OpType for Select {
    fn opname(&self) -> String {
        let Self { batch, inner, divisor, .. } = *self;
        format!("select<{batch:?}x{inner:?}, {divisor:?}>")
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.batch * self.inner, self.dtype), TType::new(self.batch, DType::I32)]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![TType::new(self.batch * (self.inner / self.divisor), self.dtype)]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
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
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = TensorOp::downcast_rc::<Self>(other) { self == other } else { false }
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
        let subdim = self.inner / self.divisor;
        let input_size = self.batch * subdim;
        let output_size = self.batch * self.inner;

        let var = match (input_size.get_var_size(input.len()), output_size.get_var_size(output.len())) {
            (None, None) => 1,
            (Some(x), Some(y)) => {
                assert_eq!(x, y);
                x
            }
            (Some(x), None) => x,
            (None, Some(x)) => x,
        };

        assert_eq!(input_size.evaluate(var), input.len());
        assert_eq!(output_size.evaluate(var), output.len());

        let batch = self.batch.evaluate(var);
        let inner = self.inner.evaluate(var);
        let subdim = subdim.evaluate(var);

        assert_eq!(input.len(), batch * subdim);
        assert_eq!(output.len(), batch * inner);
        assert_eq!(indices.len(), batch);

        for (o, index) in indices.iter().map(|&x| x as usize).enumerate() {
            for i in 0..inner {
                output[inner * o + i] = T::default()
            }

            let iidx = subdim * o;
            let oidx = inner * o + subdim * index;
            output[oidx..(subdim + oidx)].copy_from_slice(&input[iidx..(subdim + iidx)]);
        }
    }
}

impl OpType for SelectPad {
    fn opname(&self) -> String {
        let Self { batch, inner, divisor, .. } = *self;
        format!("select<{batch:?}x{inner:?}, {divisor:?}>")
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.batch * (self.inner / self.divisor), self.dtype), TType::new(self.batch, DType::I32)]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![TType::new(self.batch * self.inner, self.dtype)]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
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
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = TensorOp::downcast_rc::<Self>(other) { self == other } else { false }
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
        let batch = Size::variable();
        let inner = 16.into();
        let divisor = 4.into();

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
