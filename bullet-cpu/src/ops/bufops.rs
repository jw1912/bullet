

/// # Safety
/// Pass valid pointers and sizes.
unsafe fn buffer_operation<T: Operation>(size: usize, inp: *const f32, out: *mut f32) {
    unimplemented!();
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
        if x > 0.0 {1.0} else {0.0}
    }
}

pub(super) struct CReLU;
impl Operation for CReLU {
    fn activate(x: f32) -> f32 {
        x.clamp(0.0, 1.0)
    }

    fn prime(x: f32) -> f32 {
        if x > 0.0 && x < 1.0 {1.0} else {0.0}
    }
}

pub(super) struct SCReLU;
impl Operation for SCReLU {
    fn activate(x: f32) -> f32 {
        x.clamp(0.0, 1.0).powi(2)
    }

    fn prime(x: f32) -> f32 {
        if x > 0.0 && x < 1.0 {2.0 * x} else {0.0}
    }
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn activate_relu(size: usize, inp: *const f32, out: *mut f32) {
    buffer_operation::<ReLU>(size, inp, out);
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn activate_crelu(size: usize, inp: *const f32, out: *mut f32) {
    buffer_operation::<CReLU>(size, inp, out);
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn activate_screlu(size: usize, inp: *const f32, out: *mut f32) {
    buffer_operation::<SCReLU>(size, inp, out);
}
