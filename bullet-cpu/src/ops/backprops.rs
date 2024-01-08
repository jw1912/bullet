use super::bufops::{Operation, ReLU, CReLU, SCReLU};

/// # Safety
/// Pass valid pointers and sizes.
unsafe fn backprop_operation<T: Operation>(size: usize, inp: *const f32, out: *mut f32) {
    unimplemented!();
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn backprop_relu(size: usize, inp: *const f32, out: *mut f32) {
    backprop_operation::<ReLU>(size, inp, out);
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn backprop_crelu(size: usize, inp: *const f32, out: *mut f32) {
    backprop_operation::<CReLU>(size, inp, out);
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn backprop_screlu(size: usize, inp: *const f32, out: *mut f32) {
    backprop_operation::<SCReLU>(size, inp, out);
}
