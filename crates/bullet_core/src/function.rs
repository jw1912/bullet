mod binary;
mod matmul;
mod sparse;
mod trinary;
mod unary;

use std::fmt;

pub use binary::*;
pub use matmul::*;
pub use sparse::*;
pub use trinary::*;
pub use unary::*;

use crate::{
    device::{Device, OperationError},
    tensor::TensorRef,
};

pub trait DeviceOperation<D: Device>: 'static {
    fn opname(&self) -> String;

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>>;
}

pub struct DeviceFunction<D: Device> {
    instructions: Vec<Box<dyn DeviceOperation<D>>>,
}

impl<D: Device> Default for DeviceFunction<D> {
    fn default() -> Self {
        Self { instructions: Vec::new() }
    }
}

impl<D: Device> DeviceFunction<D> {
    pub fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        for instr in &self.instructions {
            instr.execute()?;
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, instruction: impl DeviceOperation<D>) {
        self.instructions.push(Box::new(instruction));
    }

    pub fn extend(&mut self, rhs: Self) {
        for instr in rhs.instructions {
            self.instructions.push(instr);
        }
    }
}

impl<D: Device> fmt::Display for DeviceFunction<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Instructions:")?;

        for (i, instr) in self.instructions.iter().enumerate() {
            writeln!(f)?;
            write!(f, "{i: <2}: {}", instr.opname())?;
        }

        Ok(())
    }
}

pub struct Set<D: Device> {
    pub id: TensorRef<D>,
    pub val: f32,
}

impl<D: Device> DeviceOperation<D> for Set<D> {
    fn opname(&self) -> String {
        "Set".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        self.id.borrow_mut().dense_mut()?.set_to(self.val)?;
        Ok(())
    }
}
