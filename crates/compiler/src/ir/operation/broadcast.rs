use std::rc::Rc;

use crate::{
    common::{DType, DTypeTensor, Shape, Size},
    ir::{
        IrError, IrType,
        operation::{IrOperation, IrOperationType},
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BroadcastAcrossDimension {
    dtype: DType,
    outer: Size,
    inner: Size,
    repeats: Size,
}

impl BroadcastAcrossDimension {
    pub fn new(dtype: DType, shape: impl Into<Shape>, dim: usize, repeats: impl Into<Size>) -> Result<Self, IrError> {
        let shape = shape.into();
        let shape_dim = shape.dim();
        let repeats = repeats.into();

        (dim < shape_dim)
            .then_some({
                let shape = shape.inner();
                let outer = shape[..dim].iter().cloned().reduce(|a, b| a * b).unwrap_or(1.into());
                let inner = shape[dim..].iter().cloned().reduce(|a, b| a * b).unwrap_or(1.into());

                Self { dtype, outer, inner, repeats }
            })
            .ok_or(format!("Dimension {dim} out of bounds for shape of dimension {shape_dim}!").into())
    }

    pub fn input_size(&self) -> Size {
        self.outer * self.inner
    }

    pub fn output_size(&self) -> Size {
        self.repeats * self.input_size()
    }

    pub fn apply<T: Copy + std::fmt::Debug>(&self, input: &[T], output: &mut [T]) {
        let size = input.len();

        let var = match (self.input_size().get_var_size(size), self.output_size().get_var_size(output.len())) {
            (None, None) => 1,
            (Some(x), Some(y)) => {
                assert_eq!(x, y);
                x
            }
            (Some(x), None) => x,
            (None, Some(x)) => x,
        };

        assert_eq!(self.input_size().evaluate(var), size);
        assert_eq!(self.output_size().evaluate(var), output.len());

        let outer = self.outer.evaluate(var);
        let inner = self.inner.evaluate(var);
        let repeats = self.repeats.evaluate(var);

        assert_eq!(output.len(), input.len() * repeats);

        for i in 0..outer {
            for j in 0..inner {
                let ld = inner * i;
                for r in 0..repeats {
                    output[ld * repeats + inner * r + j] = input[ld + j];
                }
            }
        }
    }
}

impl IrOperationType for BroadcastAcrossDimension {
    fn opname(&self) -> String {
        let Self { outer, inner, repeats, .. } = *self;
        format!("broadcast<{outer:?}, {inner:?}, {repeats:?}>")
    }

    fn inputs(&self) -> Vec<IrType> {
        vec![IrType::new(self.input_size(), self.dtype)]
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![IrType::new(self.output_size(), self.dtype)]
    }

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        match self.dtype {
            DType::F32 => {
                let DTypeTensor::F32(input) = inputs[0] else { panic!() };
                let DTypeTensor::F32(output) = outputs[0] else { panic!() };
                self.apply(input, output);
            }
            DType::I32 => {
                let DTypeTensor::I32(input) = inputs[0] else { panic!() };
                let DTypeTensor::I32(output) = outputs[0] else { panic!() };
                self.apply(input, output);
            }
        }
    }

    fn equals(&self, other: &Rc<dyn IrOperationType>) -> bool {
        if let Some(other) = IrOperation::downcast::<Self>(other) { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    const INPUT: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

    fn broadcast_2dim(dim: usize, expected: [i32; 16]) {
        let shape = [2, 4];

        let broadcast = BroadcastAcrossDimension::new(DType::I32, shape, dim, 2).unwrap();

        let input = DTypeTensor::I32(INPUT.to_vec());
        let mut output = DTypeTensor::I32(vec![0; 16]);

        broadcast.evaluate(&[&input], &mut [&mut output]);

        let DTypeTensor::I32(output) = output else { panic!() };

        assert_eq!(&output, &expected);
    }

    #[test]
    fn broadcast_rows() {
        broadcast_2dim(0, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn broadcast_cols() {
        broadcast_2dim(1, [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]);
    }

    fn broadcast_3dim(dim: usize, expected: [i32; 16]) {
        let shape = [2, 2, 2];

        let broadcast = BroadcastAcrossDimension::new(DType::I32, shape, dim, 2).unwrap();

        let input = DTypeTensor::I32(INPUT.to_vec());
        let mut output = DTypeTensor::I32(vec![0; 16]);

        broadcast.evaluate(&[&input], &mut [&mut output]);

        let DTypeTensor::I32(output) = output else { panic!() };

        assert_eq!(&output, &expected);
    }

    #[test]
    fn broadcast_leading_dimension() {
        broadcast_3dim(0, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn broadcast_middle_dimension() {
        broadcast_3dim(1, [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]);
    }

    #[test]
    fn broadcast_trailing_dimension() {
        broadcast_3dim(2, [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7]);
    }
}
