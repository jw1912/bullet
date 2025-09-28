use crate::device::{function::DeviceOperation, tensor::TensorRef, Device, OperationError};

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
        let mut value = self.value.dense_mut();

        value.clamp(self.min, self.max)?;

        Ok(())
    }
}