use std::sync::Arc;

use bullet_core::backend::device::Device;
use bullet_cuda_backend::CudaDevice;

fn main() {
    let device = Arc::new(CudaDevice::new(0).unwrap());
    device.sanity_check();
}
