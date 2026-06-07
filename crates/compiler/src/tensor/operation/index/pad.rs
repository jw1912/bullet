use crate::{
    ir::IRError,
    tensor::{DValue, IRTrace, OpType, Shape, Size, TNode, TType, TValue, TensorOp, operation::SliceAcrossDimension},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadAcrossDimension {
    outer: Size,
    dimen: Size,
    inner: Size,
    before: usize,
    after: usize,
    value: DValue,
}

impl PadAcrossDimension {
    pub fn outer(&self) -> Size {
        self.outer
    }

    pub fn dimen(&self) -> Size {
        self.dimen
    }

    pub fn inner(&self) -> Size {
        self.inner
    }

    pub fn before(&self) -> usize {
        self.before
    }

    pub fn after(&self) -> usize {
        self.after
    }

    pub fn value(&self) -> DValue {
        self.value
    }

    pub fn new(
        shape: impl Into<Shape>,
        dim: usize,
        before: usize,
        after: usize,
        value: DValue,
    ) -> Result<Self, IRError> {
        let shape = shape.into();
        let shape_dim = shape.dim();

        if dim >= shape_dim {
            return Err(format!("Dimension {dim} out of bounds for shape of dimension {shape_dim}!").into());
        }

        let dimen = shape[dim];
        let shape = shape.inner();
        let outer = shape[..dim].iter().cloned().reduce(|a, b| a * b).unwrap_or(1.into());
        let inner = shape[(dim + 1)..].iter().cloned().reduce(|a, b| a * b).unwrap_or(1.into());

        Ok(Self { outer, dimen, inner, before, after, value })
    }

    pub fn invert(&self) -> Result<SliceAcrossDimension, IRError> {
        let shape = [self.outer, self.before + self.dimen + self.after, self.inner];
        SliceAcrossDimension::new(self.value.dtype(), shape, 1, self.before, self.before + self.dimen.get())
    }

    pub fn input_size(&self) -> Size {
        self.outer * self.dimen * self.inner
    }

    pub fn output_size(&self) -> Size {
        self.outer * (self.before + self.dimen + self.after) * self.inner
    }

    pub fn apply<T: Copy + std::fmt::Debug>(&self, input: &[T], output: &mut [T], value: T) {
        let outer = self.outer.get();
        let inner = self.inner.get();
        let dimen = self.dimen.get();
        let middle = self.before + dimen + self.after;

        assert_eq!(input.len(), outer * inner * dimen);
        assert_eq!(output.len(), middle * outer * inner);

        for o in 0..outer {
            for i in 0..inner {
                let base = middle * inner * o;

                for b in 0..self.before {
                    output[base + inner * b + i] = value;
                }

                for d in 0..dimen {
                    output[base + inner * (self.before + d) + i] = input[inner * (dimen * o + d) + i];
                }

                for a in 0..self.after {
                    output[base + inner * (self.before + dimen + a) + i] = value;
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

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
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

        true
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(other) = other.downcast::<Self>() { self == other } else { false }
    }

    fn backward<'a>(&self, _inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let op = self.invert().unwrap();
        output_grads[0].builder().add_op([output_grads[0]], op)
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
