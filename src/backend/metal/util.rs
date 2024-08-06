/*
The things you have to do for a heterogenous interface...
*/

use metal_rs::Device;

pub use crate::backend::cpu::util::{
    calloc, copy_from_device, copy_on_device, copy_to_device, device_synchronise, free, malloc, panic_if_device_error,
    set_zero,
};

pub fn device_name() -> String {
    Device::system_default().unwrap().name().to_string()
}
