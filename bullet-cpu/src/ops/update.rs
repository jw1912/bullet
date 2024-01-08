use crate::DeviceHandles;

pub unsafe fn update_weights(
    handle: DeviceHandles,
    network_size: usize,
    decay: f32,
    adj: f32,
    rate: f32,
    network: *mut f32,
    momentum: *mut f32,
    velocity: *mut f32,
    gradients: *const f32,
) {
    unimplemented!();
}
