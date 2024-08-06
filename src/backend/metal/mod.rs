use metal_rs::{CommandQueue, Device, Library};

use crate::backend::cpu;
use ops::Kernels;

pub mod ops;
pub mod util;

#[derive(Clone)]
pub struct DeviceHandles {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
    pub(crate) library: Library,
    pub(crate) kernels: Kernels,
    pub(crate) cpu: cpu::DeviceHandles,
}

// Raw metal library file which is loaded at runtime.
const LIBRARY_SRC: &[u8] = include_bytes!("./kernels/metal.metallib");

impl Default for DeviceHandles {
    fn default() -> Self {
        let cpu = Default::default();
        let device = Device::system_default().unwrap(); // Find a device.
        let queue = device.new_command_queue();
        let library = device.new_library_with_data(LIBRARY_SRC).unwrap(); // Load the library.
        let kernels = Kernels::new(&device, &library); // Load the kernels.
        Self { device, queue, library, kernels, cpu }
    }
}

impl crate::backend::DeviceHandles {
    pub fn set_threads(&mut self, _: usize) {}
}
