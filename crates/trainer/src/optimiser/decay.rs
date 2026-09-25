use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::{
    ir::IRError,
    tensor::{DType, TType},
};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseBuilder,
    runtime::{Device, DeviceProps, Gpu, Stream},
};

use crate::optimiser::{OptimiserUpdateResult, OptimiserUpdateSync};

use super::{OptimiserState, utils::Placement};

fn build_decay_op(size: usize, decay: f32, props: &DeviceProps) -> Result<KernelSrc, IRError> {
    let p2size = if size.is_multiple_of(4) { 2 } else { 0 };
    let p2actual = 2usize.pow(u32::from(p2size));

    let builder = PointwiseBuilder::new(size / p2actual);

    let lrate_buf = builder.new_buffer(TType::new(1, DType::F32));
    let w = builder.new_buffer(TType::new(size, DType::F32));

    // the rate is the same for every weight, so it is read once as a scalar
    let lrate = lrate_buf.read(0, 0);
    let lrate = if p2size > 0 { lrate.broadcast(p2size) } else { lrate };

    let tid = builder.tid();
    let new_w = w.read(tid, p2size) * (1.0 - lrate * decay);
    w.write(tid, new_w);

    unsafe { builder.inner().lower("decay".to_string(), props) }
}

#[derive(Clone, Debug)]
pub struct WeightDecayParams<T> {
    pub inner: T,
    pub placement: Placement,
    pub decay: f32,
}

impl<T: Default> Default for WeightDecayParams<T> {
    fn default() -> Self {
        Self { inner: T::default(), placement: Placement::Before, decay: 0.01 }
    }
}

pub struct WeightDecay<G: Gpu, S: OptimiserState<G>> {
    inner: S,
    placement: Placement,
    op: CompiledKernel<G>,
    device: Arc<Device<G>>,
    size: usize,
}

impl<G: Gpu, S: OptimiserState<G>> OptimiserState<G> for WeightDecay<G, S> {
    type Params = WeightDecayParams<S::Params>;

    fn new(device: &Arc<Device<G>>, size: usize, params: Self::Params) -> Result<Self, G::Error> {
        Ok(Self {
            op: build_decay_op(size, params.decay, device.props()).unwrap().compile(device.clone())?,
            inner: S::new(device, size, params.inner.clone())?,
            placement: params.placement,
            device: device.clone(),
            size,
        })
    }

    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
        let mut blocks = OptimiserUpdateSync::default();

        let rate = vec![learning_rate.clone()];

        if self.placement == Placement::Before {
            blocks.push_kernel(self.op.execute(stream.clone(), rate, vec![weights.clone()])?);
            blocks.extend_by(self.inner.update(stream, weights, grads, gradient_factor, learning_rate)?);
        } else {
            blocks.extend_by(self.inner.update(stream, weights.clone(), grads, gradient_factor, learning_rate)?);
            blocks.push_kernel(self.op.execute(stream.clone(), rate, vec![weights])?);
        }

        Ok(blocks)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        self.inner.reset()
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        self.inner.set_params(params.inner)?;
        self.placement = params.placement;

        self.op = build_decay_op(self.size, params.decay, self.device.props()).unwrap().compile(self.device.clone())?;
        Ok(())
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.inner)).collect();
        S::load_from_checkpoint(&mut map, path)
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.inner)).collect();
        S::write_to_checkpoint(&map, path)
    }
}
