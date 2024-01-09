use crate::DeviceHandles;

const B1: f32 = 0.9;
const B2: f32 = 0.999;
const B1P: f32 = 1.0 - B1;
const B2P : f32= 1.0 - B2;
const EPSILON: f32 = 0.00000001;
const MAX: f32 = 1.98;

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
    let network = network as usize;
    let momentum = momentum as usize;
    let velocity = velocity as usize;
    let gradients = gradients as usize;

    handle.split_workload(network_size, |_, idx| {
        let grad = adj * *(gradients as *const f32).add(idx);
        let p = (network as *mut f32).add(idx);
        let m = (momentum as *mut f32).add(idx);
        let v = (velocity as *mut f32).add(idx);

        let mut param = *p * decay;

        *m = B1 * *m + B1P * grad;
        *v = B2 * *v + B2P * grad * grad;

        param -= rate * *m / ((*v).sqrt() + EPSILON);
        param = param.clamp(-MAX, MAX);

        *p = param;
    });
}
