use crate::DeviceHandles;

use super::bufops::{Operation, ReLU, CReLU, SCReLU};

unsafe fn backprop_operation<T: Operation>(handle: DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    unimplemented!();
}

pub unsafe fn backprop_relu(handle: DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    backprop_operation::<ReLU>(handle, size, inp, out);
}

pub unsafe fn backprop_crelu(handle: DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    backprop_operation::<CReLU>(handle, size, inp, out);
}

pub unsafe fn backprop_screlu(handle: DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    backprop_operation::<SCReLU>(handle, size, inp, out);
}
