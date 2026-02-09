use std::rc::Rc;

use crate::ir::{
    graph::{DType, DValue, GraphError, Op, OpType, Shape, Size, TType, TValue},
    operation::PadAcrossDimension,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceAcrossDimension {
    dtype: DType,
    outer: Size,
    dimen: usize,
    inner: Size,
    start: usize,
    end: usize,
}

impl SliceAcrossDimension {
    pub fn new(
        dtype: DType,
        shape: impl Into<Shape>,
        dim: usize,
        start: usize,
        end: usize,
    ) -> Result<Self, GraphError> {
        let shape = shape.into();
        let shape_dim = shape.dim();

        if dim >= shape_dim {
            return Err(format!("Dimension {dim} out of bounds for shape of dimension {shape_dim}!").into());
        }

        let dimen = shape[dim].evaluate_constant().ok_or("Slice does not support variable size slice dim!")?;

        if end <= start || end > dimen {
            return Err("Invalid slice params!".into());
        }

        let shape = shape.inner();
        let outer = shape[..dim].iter().cloned().reduce(|a, b| a * b).unwrap_or(1.into());
        let inner = shape[(dim + 1)..].iter().cloned().reduce(|a, b| a * b).unwrap_or(1.into());

        Ok(Self { dtype, outer, dimen, inner, start, end })
    }

    pub fn invert(&self) -> Result<PadAcrossDimension, GraphError> {
        let shape = [self.outer, self.dimen.into(), self.inner];
        PadAcrossDimension::new(shape, 1, self.start, self.end, DValue::zero(self.dtype))
    }

    pub fn input_size(&self) -> Size {
        self.outer * self.dimen * self.inner
    }

    pub fn output_size(&self) -> Size {
        self.outer * (self.end - self.start) * self.inner
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

        assert_eq!(output.len(), outer * (self.end - self.start) * inner);

        let odimen = self.end - self.start;

        for o in 0..outer {
            for d in 0..odimen {
                let oidx = inner * odimen * o + inner * d;
                let iidx = inner * self.dimen * o + inner * (d + self.start);
                output[oidx..(inner + oidx)].copy_from_slice(&input[iidx..(inner + iidx)]);
            }
        }
    }
}

impl OpType for SliceAcrossDimension {
    fn opname(&self) -> String {
        let Self { outer, dimen, inner, start, end, .. } = *self;
        format!("slice<{outer:?}x{dimen}x{inner:?}, {start}, {end}>")
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.input_size(), self.dtype)]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![TType::new(self.output_size(), self.dtype)]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        match self.dtype {
            DType::F32 => {
                let TValue::F32(input) = inputs[0] else { panic!() };
                let TValue::F32(output) = &mut outputs[0] else { panic!() };
                self.apply(input, output);
            }
            DType::I32 => {
                let TValue::I32(input) = inputs[0] else { panic!() };
                let TValue::I32(output) = &mut outputs[0] else { panic!() };
                self.apply(input, output);
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

    fn slice_2dim(dim: usize, expected: Vec<i32>) {
        let shape = [2, 4];

        let slice = SliceAcrossDimension::new(DType::I32, shape, dim, 0, 1).unwrap();

        let input = TValue::I32(INPUT.to_vec());
        let mut output = TValue::I32(vec![0; expected.len()]);

        slice.evaluate(vec![&input], vec![&mut output]);

        let TValue::I32(output) = output else { panic!() };

        assert_eq!(&output, &expected);
    }

    #[test]
    fn slice_rows() {
        slice_2dim(0, vec![0, 1, 2, 3]);
    }

    #[test]
    fn slice_cols() {
        slice_2dim(1, vec![0, 4]);
    }

    fn slice_3dim(dim: usize, expected: Vec<i32>) {
        let shape = [2, 2, 2];

        let slice = SliceAcrossDimension::new(DType::I32, shape, dim, 0, 1).unwrap();

        let input = TValue::I32(INPUT.to_vec());
        let mut output = TValue::I32(vec![0; expected.len()]);

        slice.evaluate(vec![&input], vec![&mut output]);

        let TValue::I32(output) = output else { panic!() };

        assert_eq!(&output, &expected);
    }

    #[test]
    fn slice_leading_dimension() {
        slice_3dim(0, vec![0, 1, 2, 3]);
    }

    #[test]
    fn slice_middle_dimension() {
        slice_3dim(1, vec![0, 1, 4, 5]);
    }

    #[test]
    fn slice_trailing_dimension() {
        slice_3dim(2, vec![0, 2, 4, 6]);
    }
}
