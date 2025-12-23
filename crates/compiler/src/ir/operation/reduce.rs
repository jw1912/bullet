use std::{cmp::Ordering, ops::Add, rc::Rc};

use crate::{
    common::{DType, DTypeTensor, Shape, Size},
    ir::{IrError, IrType, operation::IrOperation},
};

use super::IrOperationType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    Sum,
    Max,
    Min,
}

impl Reduction {
    pub fn apply<T>(self, lhs: T, rhs: T) -> T
    where
        T: Add<T, Output = T> + Copy + PartialOrd,
    {
        match self {
            Self::Sum => lhs + rhs,
            Self::Max => match lhs.partial_cmp(&rhs).expect("Not comparable!") {
                Ordering::Equal => lhs,
                Ordering::Greater => lhs,
                Ordering::Less => rhs,
            },
            Self::Min => match lhs.partial_cmp(&rhs).expect("Not comparable!") {
                Ordering::Equal => lhs,
                Ordering::Greater => rhs,
                Ordering::Less => lhs,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReduceAcrossDimension {
    dtype: DType,
    outer: Size,
    dimen: Size,
    inner: Size,
    reduction: Reduction,
}

impl ReduceAcrossDimension {
    pub fn new(dtype: DType, shape: impl Into<Shape>, dim: usize, reduction: Reduction) -> Result<Self, IrError> {
        let shape = shape.into();
        let shape_dim = shape.dim();

        (dim < shape_dim)
            .then_some({
                let shape = shape.inner();
                let outer = shape[..dim].iter().cloned().reduce(|a, b| a * b).unwrap_or(1.into());
                let inner = shape[(dim + 1)..].iter().cloned().reduce(|a, b| a * b).unwrap_or(1.into());
                let dimen = shape[dim];

                Self { dtype, outer, dimen, inner, reduction }
            })
            .ok_or(format!("Dimension {dim} out of bounds for shape of dimension {shape_dim}!").into())
    }

    pub fn input_size(&self) -> Size {
        self.outer * self.dimen * self.inner
    }

    pub fn apply<T>(&self, input: &[T], output: &mut [T])
    where
        T: Add<T, Output = T> + Copy + Default + PartialOrd,
    {
        let size = input.len();
        let var = self.input_size().get_var_size(size).unwrap_or(1);

        let outer = self.outer.evaluate(var);
        let dimen = self.dimen.evaluate(var);
        let inner = self.inner.evaluate(var);

        assert_eq!(output.len() * dimen, input.len());

        let outer_stride = dimen * inner;

        for i in 0..outer {
            for j in 0..inner {
                let mut acc = input[i * outer_stride + j];

                for k in 1..dimen {
                    acc = self.reduction.apply(acc, input[i * outer_stride + k * inner + j]);
                }

                output[i * inner + j] = acc;
            }
        }
    }
}

impl IrOperationType for ReduceAcrossDimension {
    fn opname(&self) -> String {
        format!("reduce.{:?}<{:?}, {:?}, {:?}>", self.reduction, self.outer, self.dimen, self.inner)
    }

    fn inputs(&self) -> Vec<IrType> {
        vec![IrType::new(self.input_size(), self.dtype)]
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![IrType::new(self.input_size() / self.dimen, self.dtype)]
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
    const INPUT: [i32; 64] = [
         0,  1,  2,  3,  4,  5,  6,  7,
         8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63,
    ];

    fn reduce_dim<const M: usize>(dim: usize, reduction: Reduction, expected: impl Fn(usize) -> usize) {
        let shape = [8, 2, 4];

        let reduction = ReduceAcrossDimension::new(DType::I32, shape, dim, reduction).unwrap();

        let input = DTypeTensor::I32(INPUT.to_vec());
        let mut output = DTypeTensor::I32(vec![0; 64 / shape[dim]]);

        reduction.evaluate(&[&input], &mut [&mut output]);

        let DTypeTensor::I32(output) = output else { panic!() };

        let expected = std::array::from_fn::<_, M, _>(|i| expected(i) as i32);

        assert_eq!(&output, &expected);
    }

    #[test]
    fn reduce_sum_leading_dimension() {
        reduce_dim::<8>(0, Reduction::Sum, |i| 224 + 8 * i);
    }

    #[test]
    fn reduce_sum_middle_dimension() {
        reduce_dim::<32>(1, Reduction::Sum, |i| 4 + 2 * (i % 4) + 16 * (i / 4));
    }

    #[test]
    fn reduce_sum_trailing_dimension() {
        reduce_dim::<16>(2, Reduction::Sum, |i| 6 + 16 * i);
    }

    #[test]
    fn reduce_max_leading_dimension() {
        reduce_dim::<8>(0, Reduction::Max, |i| 56 + i);
    }

    #[test]
    fn reduce_max_middle_dimension() {
        reduce_dim::<32>(1, Reduction::Max, |i| i + 4 * (1 + i / 4));
    }

    #[test]
    fn reduce_max_trailing_dimension() {
        reduce_dim::<16>(2, Reduction::Max, |i| 3 + 4 * i);
    }

    #[test]
    fn reduce_min_leading_dimension() {
        reduce_dim::<8>(0, Reduction::Min, |i| i);
    }

    #[test]
    fn reduce_min_middle_dimension() {
        reduce_dim::<32>(1, Reduction::Min, |i| i + 4 * (i / 4));
    }

    #[test]
    fn reduce_min_trailing_dimension() {
        reduce_dim::<16>(2, Reduction::Min, |i| 4 * i);
    }
}
