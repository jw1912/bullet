use crate::{backend::{ops, util, DeviceHandles}, DeviceBuffer};

/// A struct intended to hold all network weights and biases
/// needed for training.
pub struct Optimiser {
    size: usize,
    network: DeviceBuffer,
    momentum: DeviceBuffer,
    velocity: DeviceBuffer,
    gradients: DeviceBuffer,
}

impl Optimiser {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            network: DeviceBuffer::new(size),
            momentum: DeviceBuffer::new(size),
            velocity: DeviceBuffer::new(size),
            gradients: DeviceBuffer::new(size),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn zero_gradient(&self) {
        util::set_zero(self.gradients.ptr(), self.gradients.size());
    }

    /// Pointer to network buffer starting at `network.ptr() + index`.
    pub fn weights_offset(&self, index: usize) -> *mut f32 {
        assert!(
            index < self.size,
            "Index out of bounds: {index} >= {}!",
            self.size
        );
        unsafe { self.network.ptr().add(index) }
    }

    /// Pointer to gradient buffer starting at `gradient.ptr() + index`.
    pub fn gradients_offset(&self, index: usize) -> *mut f32 {
        assert!(
            index < self.size,
            "Index out of bounds: {index} >= {}!",
            self.size
        );
        unsafe { self.gradients.ptr().add(index) }
    }

    pub fn update(&self, handle: DeviceHandles, decay: f32, adj: f32, rate: f32) {
        let decay_gamma = 1.0 - decay * rate;
        unsafe {
            ops::update_weights(
                handle,
                self.size,
                decay_gamma,
                adj,
                rate,
                self.network.ptr(),
                self.momentum.ptr(),
                self.velocity.ptr(),
                self.gradients.ptr(),
            );
        }
    }

    pub fn load_weights_from_host(&self, network: &[f32]) {
        self.network.load_from_host(network);
    }

    pub fn load_from_cpu(&self, network: &[f32], momentum: &[f32], velocity: &[f32]) {
        self.network.load_from_host(network);
        self.momentum.load_from_host(momentum);
        self.velocity.load_from_host(velocity);
    }

    pub fn write_weights_to_host(&self, buf: &mut [f32]) {
        self.network.write_to_host(buf);
    }

    pub fn write_to_host(&self, network: &mut [f32], momentum: &mut [f32], velocity: &mut [f32]) {
        self.network.write_to_host(network);
        self.momentum.write_to_host(momentum);
        self.velocity.write_to_host(velocity);
    }
}
