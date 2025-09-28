use crate::device::{Device, OperationError, function::DeviceOperation, tensor::TensorRef};

#[derive(Clone)]
pub struct ClipInPlace<D: Device> {
    pub value: TensorRef<D>,
    pub min: f32,
    pub max: f32,
}

impl<D: Device> DeviceOperation<D> for ClipInPlace<D> {
    fn opname(&self) -> String {
        "ClipInPlace".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        self.value.dense_mut().clamp(self.min, self.max)
    }
}
