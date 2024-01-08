use crate::DeviceHandles;

pub unsafe fn sigmoid_mse(
    handle: DeviceHandles,
    buffer_size: usize,
    outputs: *mut f32,
    results: *const f32,
    error: *mut f32,
) {
    unimplemented!();
}
