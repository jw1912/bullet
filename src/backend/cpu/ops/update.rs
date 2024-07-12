use super::DeviceHandles;

const EPSILON: f32 = 0.00000001;

pub unsafe fn update_weights(
    handle: &DeviceHandles,
    network_size: usize,
    decay: f32,
    beta1: f32,
    beta2: f32,
    min_weight: f32,
    max_weight: f32,
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

        *m = beta1 * *m + (1.0 - beta1) * grad;
        *v = beta2 * *v + (1.0 - beta2) * grad * grad;

        param -= rate * *m / ((*v).sqrt() + EPSILON);
        param = param.clamp(min_weight, max_weight);

        *p = param;
    });
}
