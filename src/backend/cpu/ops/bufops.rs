use super::DeviceHandles;

unsafe fn buffer_operation<T: Operation>(handle: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    let inp = inp as usize;
    let out = out as usize;

    handle.split_workload(size, |_, idx| {
        let this_inp = (inp as *const f32).add(idx);
        let this_out = (out as *mut f32).add(idx);
        *this_out = T::activate(*this_inp);
    });
}

pub(super) trait Operation {
    fn activate(x: f32) -> f32;

    fn prime(x: f32) -> f32;
}

pub(super) struct ReLU;

impl Operation for ReLU {
    fn activate(x: f32) -> f32 {
        x.max(0.0)
    }

    fn prime(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub(super) struct CReLU;

impl Operation for CReLU {
    fn activate(x: f32) -> f32 {
        x.clamp(0.0, 1.0)
    }

    fn prime(x: f32) -> f32 {
        if x > 0.0 && x < 1.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub(super) struct SCReLU;

impl Operation for SCReLU {
    fn activate(x: f32) -> f32 {
        x.clamp(0.0, 1.0).powi(2)
    }

    fn prime(x: f32) -> f32 {
        if x > 0.0 && x < 1.0 {
            2.0 * x
        } else {
            0.0
        }
    }
}

pub unsafe fn activate_relu(handle: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    buffer_operation::<ReLU>(handle, size, inp, out);
}

pub unsafe fn activate_crelu(handle: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    buffer_operation::<CReLU>(handle, size, inp, out);
}

pub unsafe fn activate_screlu(handle: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    buffer_operation::<SCReLU>(handle, size, inp, out);
}

pub unsafe fn add_to(handle: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
    let inp = inp as usize;
    let out = out as usize;

    handle.split_workload(size, |_, idx| {
        let this_inp = (inp as *const f32).add(idx);
        let this_out = (out as *mut f32).add(idx);
        *this_out += *this_inp;
    });
}
