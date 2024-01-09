use crate::DeviceHandles;

use super::bufops::{Operation, ReLU, CReLU, SCReLU};

unsafe fn backprop_operation<T: Operation>(handle: DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    let inp = inp as usize;
    let out = out as usize;

    handle.split_workload(size, |_, idx| {
        let this_inp = (inp as *const f32).add(idx);
        let this_out = (out as *mut f32).add(idx);
        *this_out = *this_inp * T::prime(*this_out);
    });
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
