use metal_rs::{Device, Library};

use ops::Kernels;

pub mod ops;
pub mod util;

#[derive(Clone)]
pub struct DeviceHandles {
    pub(crate) device: Device,
    pub(crate) library: Library,
    pub(crate) kernels: Kernels,
}

const LIBRARY_SRC: &[u8] = include_bytes!("./kernels/metal.metallib");

impl Default for DeviceHandles {
    fn default() -> Self {
        let device = Device::system_default().unwrap();
        let library = device.new_library_with_data(LIBRARY_SRC).unwrap();
        let kernels = Kernels::new(&library);
        Self { device, library, kernels }
    }
}

impl crate::backend::DeviceHandles {
    pub fn set_threads(&mut self, _: usize) {}
}
