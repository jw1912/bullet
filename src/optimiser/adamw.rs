use super::{Optimiser, OptimiserBase, OptimiserType};
use crate::backend::{ops, DeviceHandles};
use crate::tensor::DeviceBuffer;
use crate::util;

#[derive(Default)]
pub struct AdamW;

impl OptimiserType for AdamW {
    type Optimiser = AdamWOptimiser;
}

#[derive(Clone, Debug)]
pub struct AdamWParams {
    pub decay: f32,
}

pub struct AdamWOptimiser {
    base: OptimiserBase,
    momentum: DeviceBuffer,
    velocity: DeviceBuffer,
}

impl Optimiser for AdamWOptimiser {
    type AdditionalOptimiserParams = AdamWParams;

    fn new(size: usize) -> Self {
        Self { base: OptimiserBase::new(size), momentum: DeviceBuffer::new(size), velocity: DeviceBuffer::new(size) }
    }

    fn size(&self) -> usize {
        self.base.size
    }

    fn zero_gradient(&self) {
        self.base.zero_gradient()
    }

    fn weights_offset(&self, index: usize) -> *mut f32 {
        self.base.weights_offset(index)
    }

    fn gradients_offset(&self, index: usize) -> *mut f32 {
        self.base.gradients_offset(index)
    }

    fn load_weights_from_host(&self, network: &[f32]) {
        self.base.load_weights_from_host(network)
    }

    fn write_weights_to_host(&self, buf: &mut [f32]) {
        self.base.write_weights_to_host(buf);
    }

    fn update(&self, handle: &DeviceHandles, grad_adj: f32, lr: f32, params: &AdamWParams) {
        let decay_gamma = 1.0 - params.decay * lr;
        unsafe {
            ops::update_weights(
                handle,
                self.base.size(),
                decay_gamma,
                grad_adj,
                lr,
                self.base.network.ptr(),
                self.momentum.ptr(),
                self.velocity.ptr(),
                self.base.gradients.ptr(),
            );
        }
    }

    fn write_to_checkpoint(&self, path: &str) {
        let size = self.base.size();

        let mut buf1 = vec![0.0; size];
        let mut buf2 = vec![0.0; size];
        let mut buf3 = vec![0.0; size];

        self.base.network.write_to_host(&mut buf1);
        self.momentum.write_to_host(&mut buf2);
        self.velocity.write_to_host(&mut buf3);

        util::write_to_bin(&buf1, size, &format!("{path}/params.bin"), false)
            .unwrap_or_else(|_| panic!("Writing to [{path}/params.bin] failed!"));
        util::write_to_bin(&buf2, size, &format!("{path}/momentum.bin"), false)
            .unwrap_or_else(|_| panic!("Writing to [{path}/momentum.bin] failed!"));
        util::write_to_bin(&buf3, size, &format!("{path}/velocity.bin"), false)
            .unwrap_or_else(|_| panic!("Writing to [{path}/velocity.bin] failed!"));
    }

    fn load_from_checkpoint(&self, path: &str) {
        let size = self.base.size();

        let network = util::load_from_bin_f32_slice(size, format!("{path}/params.bin").as_str());
        let momentum = util::load_from_bin_f32_slice(size, format!("{path}/momentum.bin").as_str());
        let velocity = util::load_from_bin_f32_slice(size, format!("{path}/velocity.bin").as_str());

        self.base.network.load_from_host(&network);
        self.momentum.load_from_host(&momentum);
        self.velocity.load_from_host(&velocity);
    }
}
