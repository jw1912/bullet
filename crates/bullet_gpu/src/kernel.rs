pub mod args;
pub mod expr;

use bullet_core::device::Device;

pub trait RuntimeKernelSupport: Device {
    type Function;
    type Builder;
}
