pub mod bwd;
pub mod concat;
pub mod fwd;
pub mod linear_comb;
pub mod matmul;
pub mod slice;
pub mod sparse;

use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::backend::{device::Device, tensor::DenseMatrix};

pub fn setup_ones<D: Device>(
    device: Arc<D>,
    internal: &mut HashMap<String, RefCell<DenseMatrix<D>>>,
    batch_size: usize,
) -> Result<(), D::DeviceError> {
    if let Some(ones) = internal.get_mut("ones") {
        if ones.borrow().size() < batch_size {
            *ones = RefCell::new(DenseMatrix::ones(device, batch_size)?);
        }
    } else {
        let ones = RefCell::new(DenseMatrix::ones(device, batch_size)?);
        internal.insert("ones".to_string(), ones);
    }

    Ok(())
}

pub fn setup_softmax<D: Device>(
    device: Arc<D>,
    internal: &mut HashMap<String, RefCell<DenseMatrix<D>>>,
    size: usize,
) -> Result<(), D::DeviceError> {
    if !internal.contains_key("softmaxed") {
        let zeros = RefCell::new(DenseMatrix::zeroed(device.clone(), size)?);
        internal.insert("softmaxed".to_string(), zeros);
        let zeros = RefCell::new(DenseMatrix::zeroed(device, size)?);
        internal.insert("individual_losses".to_string(), zeros);
    }

    Ok(())
}
