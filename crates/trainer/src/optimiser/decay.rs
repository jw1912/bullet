use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::{
    ir::IRError,
    tensor::{DType, DValue, TType, operation::CABinary},
};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseIR,
    runtime::{Device, Gpu, Stream},
};

use crate::optimiser::{OptimiserUpdateResult, OptimiserUpdateSync};

use super::{OptimiserState, utils::Placement};

fn build_decay_op(size: usize, decay: f32) -> Result<KernelSrc, IRError> {
    let mut pntwise = PointwiseIR::new(size.into())?;

    let zero = pntwise.add_const(DValue::I32(0), 0);
    let one = pntwise.add_const(DValue::F32(1.0), 0);
    let neg = pntwise.add_const(DValue::F32(-1.0), 0);

    let decay = pntwise.add_const(DValue::F32(decay), 0);
    let lrate = pntwise.add_buf(TType::new(1, DType::F32));
    let lrate = pntwise.read(lrate, zero, 0)?;

    let w = pntwise.add_buf(TType::new(size, DType::F32));
    let old_w = pntwise.read(w, pntwise.tid(), 0)?;

    let amt = pntwise.binary(lrate, decay, CABinary::Mul)?;
    let neg = pntwise.binary(neg, amt, CABinary::Mul)?;
    let fac = pntwise.binary(one, neg, CABinary::Add)?;
    let new_w = pntwise.binary(old_w, fac, CABinary::Mul)?;

    pntwise.write(w, pntwise.tid(), new_w)?;

    unsafe { pntwise.lower("decay".to_string()) }
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
            op: build_decay_op(size, params.decay).unwrap().compile(device.clone())?,
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

        if self.placement == Placement::Before {
            blocks.push_kernel(self.op.execute(stream.clone(), Vec::new(), vec![weights.clone()])?);
            blocks.extend_by(self.inner.update(stream, weights, grads, gradient_factor, learning_rate)?);
        } else {
            blocks.extend_by(self.inner.update(stream, weights.clone(), grads, gradient_factor, learning_rate)?);
            blocks.push_kernel(self.op.execute(stream.clone(), Vec::new(), vec![weights])?);
        }

        Ok(blocks)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        self.inner.reset()
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        self.inner.set_params(params.inner)?;
        self.placement = params.placement;

        self.op = build_decay_op(self.size, params.decay).unwrap().compile(self.device.clone())?;
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
