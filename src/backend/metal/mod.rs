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

// Raw metal library file which is loaded at runtime.
const LIBRARY_SRC: &[u8] = include_bytes!("./kernels/metal.metallib");

impl Default for DeviceHandles {
    fn default() -> Self {
        let device = Device::system_default().unwrap(); // Find a device.
        let library = device.new_library_with_data(LIBRARY_SRC).unwrap(); // Load the library.
        let kernels = Kernels::new(&library); // Load the kernels.
        Self { device, library, kernels }
    }
}

impl crate::backend::DeviceHandles {
    pub fn set_threads(&mut self, _: usize) {}
}
