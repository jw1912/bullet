mod adamw;

pub use adamw::{AdamW, AdamWParams};

use crate::{
    backend::{util, DeviceHandles},
    tensor::DeviceBuffer,
};

pub struct OptimiserBase {
    size: usize,
    network: DeviceBuffer,
    gradients: DeviceBuffer,
}

impl OptimiserBase {
    pub fn new(size: usize) -> Self {
        Self { size, network: DeviceBuffer::new(size), gradients: DeviceBuffer::new(size) }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn zero_gradient(&self) {
        util::set_zero(self.gradients.ptr(), self.gradients.size());
    }

    /// Pointer to network buffer starting at `network.ptr() + index`.
    pub fn weights_offset(&self, index: usize) -> *mut f32 {
        assert!(index < self.size, "Index out of bounds: {index} >= {}!", self.size);
        unsafe { self.network.ptr().add(index) }
    }

    /// Pointer to gradient buffer starting at `gradient.ptr() + index`.
    pub fn gradients_offset(&self, index: usize) -> *mut f32 {
        assert!(index < self.size, "Index out of bounds: {index} >= {}!", self.size);
        unsafe { self.gradients.ptr().add(index) }
    }

    pub fn load_weights_from_host(&self, network: &[f32]) {
        self.network.load_from_host(network);
    }

    pub fn write_weights_to_host(&self, buf: &mut [f32]) {
        self.network.write_to_host(buf);
    }
}

pub trait OptimiserType: Default {
    type Optimiser: Optimiser;
}

pub trait Optimiser {
    type AdditionalOptimiserParams: Clone + std::fmt::Debug + Send + Sync;

    fn new(size: usize) -> Self;

    fn size(&self) -> usize;

    fn zero_gradient(&self);

    /// Pointer to network buffer starting at `network.ptr() + index`.
    fn weights_offset(&self, index: usize) -> *mut f32;

    /// Pointer to gradient buffer starting at `gradient.ptr() + index`.
    fn gradients_offset(&self, index: usize) -> *mut f32;

    fn update(&self, handle: DeviceHandles, grad_adj: f32, lr: f32, params: &Self::AdditionalOptimiserParams);

    fn load_weights_from_host(&self, network: &[f32]);

    fn write_weights_to_host(&self, buf: &mut [f32]);

    fn load_from_checkpoint(&self, path: &str);

    fn write_to_checkpoint(&self, path: &str);
}
