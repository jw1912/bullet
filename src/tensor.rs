pub(crate) mod buffer;
mod operations;
mod raw_tensor;
mod shape;

pub use operations::Operation;
use raw_tensor::RawTensor;
pub use shape::Shape;

use crate::backend::ExecutionContext;

impl From<Tensor> for Shape {
    fn from(value: Tensor) -> Self {
        value.values.shape()
    }
}

#[derive(Debug, Default)]
pub struct Tensor {
    pub(crate) values: RawTensor,
    pub(crate) gradients: Option<RawTensor>,
}

impl diffable::Tensor for Tensor {
    type ModelOfTensor = Shape;
    type ExecutionContext = ExecutionContext;

    fn new(desc: Self::ModelOfTensor, requires_grad: bool) -> Self {
        Self {
            values: RawTensor::new_empty(desc),
            gradients: if requires_grad {
                Some(RawTensor::new_empty(desc))
            } else {
                None
            },
        }
    }

    fn zero_grad(&mut self) {
        if let Some(grad) = self.gradients.as_mut() {
            grad.set_zero();
        }
    }

    fn copy_values_into(&self, dest: &mut Self) {
        self.values.copy_values_into(&mut dest.values);
    }

    fn get_scalar(&self) -> Option<f32> {
        if self.values.is_scalar() {
            let mut buf = [0.0];
            self.values.write_dense_to_slice(&mut buf);
            Some(buf[0])
        } else {
            None
        }
    }

    fn set_grad_to_unit(&mut self) {
        let grad = self.gradients.as_mut().unwrap();
        assert!(grad.is_scalar());
        grad.load_dense_from_slice(&[1.0]);
    }
}
