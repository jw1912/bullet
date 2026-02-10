use std::rc::Rc;

use crate::ir::{
    graph::{DValue, GraphError, Op, OpType, Shape, Size, TType, TValue},
    operation::SliceAcrossDimension,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadAcrossDimension {
    outer: Size,
    dimen: usize,
    inner: Size,
    before: usize,
    after: usize,
    value: DValue,
}

impl PadAcrossDimension {
    pub fn new(
        shape: impl Into<Shape>,
        dim: usize,
        before: usize,
        after: usize,
        value: DValue,
    ) -> Result<Self, GraphError> {
        let shape = shape.into();
        let shape_dim = shape.dim();

        if dim >= shape_dim {
            return Err(format!("Dimension {dim} out of bounds for shape of dimension {shape_dim}!").into());
        }

        let dimen = shape[dim].evaluate_constant().ok_or("Slice does not support variable size slice dim!")?;

        let shape = shape.inner();
        let outer = shape[..dim].iter().cloned().reduce(|a, b| a * b).unwrap_or(1.into());
        let inner = shape[(dim + 1)..].iter().cloned().reduce(|a, b| a * b).unwrap_or(1.into());

        Ok(Self { outer, dimen, inner, before, after, value })
    }

    pub fn invert(&self) -> Result<SliceAcrossDimension, GraphError> {
        let shape = [self.outer, (self.before + self.dimen + self.after).into(), self.inner];
        SliceAcrossDimension::new(self.value.dtype(), shape, 1, self.before, self.before + self.dimen)
    }

    pub fn input_size(&self) -> Size {
        self.outer * self.dimen * self.inner
    }

    pub fn output_size(&self) -> Size {
        self.outer * (self.before + self.dimen + self.after) * self.inner
    }

    pub fn apply<T: Copy + std::fmt::Debug>(&self, input: &[T], output: &mut [T], value: T) {
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

        assert_eq!(output.len(), input.len() + (self.before + self.after) * outer * inner);

        let middle = self.before + self.dimen + self.after;

        for o in 0..outer {
            for i in 0..inner {
                let base = middle * inner * o;

                for b in 0..self.before {
                    output[base + inner * b + i] = value;
                }

                for d in 0..self.dimen {
                    output[base + inner * (self.before + d) + i] = input[inner * (self.dimen * o + d) + i];
                }

                for a in 0..self.after {
                    output[base + inner * (self.before + self.dimen + a) + i] = value;
                }
            }
        }
    }
}

impl OpType for PadAcrossDimension {
    fn opname(&self) -> String {
        let Self { outer, dimen, inner, before, after, value } = *self;
        format!("pad<{outer:?}x{dimen:?}x{inner:?}, {before:?}, {after:?}, {value}>")
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.input_size(), self.value.dtype())]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![TType::new(self.output_size(), self.value.dtype())]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        match self.value {
            DValue::F32(value) => {
                let TValue::F32(input) = inputs[0] else { panic!() };
                let TValue::F32(output) = &mut outputs[0] else { panic!() };
                self.apply(input, output, value);
            }
            DValue::I32(value) => {
                let TValue::I32(input) = inputs[0] else { panic!() };
                let TValue::I32(output) = &mut outputs[0] else { panic!() };
                self.apply(input, output, value);
            }
        }
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = Op::downcast_rc::<Self>(other) { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    const INPUT: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

    fn pad_2dim(dim: usize, expected: Vec<i32>) {
        let shape = [2, 4];

        let pad = PadAcrossDimension::new(shape, dim, 1, 1, 0.into()).unwrap();

        let input = TValue::I32(INPUT.to_vec());
        let mut output = TValue::I32(vec![0; expected.len()]);

        pad.evaluate(vec![&input], vec![&mut output]);

        let TValue::I32(output) = output else { panic!() };

        assert_eq!(&output, &expected);

        let inverse = pad.invert().unwrap();
        assert_eq!(pad, inverse.invert().unwrap());
        let mut inverse_output = TValue::I32(vec![0; INPUT.len()]);
        inverse.evaluate(vec![&TValue::I32(output)], vec![&mut inverse_output]);
        let TValue::I32(inverse_output) = inverse_output else { panic!() };
        assert_eq!(&INPUT[..], &inverse_output[..]);
    }

    #[test]
    fn pad_rows() {
        pad_2dim(0, vec![0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0]);
    }

    #[test]
    fn pad_cols() {
        pad_2dim(1, vec![0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 7, 0]);
    }

    fn pad_3dim(dim: usize, expected: Vec<i32>) {
        let shape = [2, 2, 2];

        let pad = PadAcrossDimension::new(shape, dim, 1, 1, 0.into()).unwrap();

        let input = TValue::I32(INPUT.to_vec());
        let mut output = TValue::I32(vec![0; expected.len()]);

        pad.evaluate(vec![&input], vec![&mut output]);

        let TValue::I32(output) = output else { panic!() };

        assert_eq!(&output, &expected);

        let inverse = pad.invert().unwrap();
        assert_eq!(pad, inverse.invert().unwrap());
        let mut inverse_output = TValue::I32(vec![0; INPUT.len()]);
        inverse.evaluate(vec![&TValue::I32(output)], vec![&mut inverse_output]);
        let TValue::I32(inverse_output) = inverse_output else { panic!() };
        assert_eq!(&INPUT[..], &inverse_output[..]);
    }

    #[test]
    fn pad_leading_dimension() {
        pad_3dim(0, vec![0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0]);
    }

    #[test]
    fn pad_middle_dimension() {
        pad_3dim(1, vec![0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 7, 0, 0]);
    }

    #[test]
    fn pad_trailing_dimension() {
        pad_3dim(2, vec![0, 0, 1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0, 6, 7, 0]);
    }
}
