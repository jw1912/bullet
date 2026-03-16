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

use crate::optimiser::OptimiserUpdateSync;

use super::{OptimiserState, OptimiserUpdateResult, utils::Placement};

fn build_clip_op(size: usize, min: f32, max: f32) -> Result<KernelSrc, IRError> {
    let mut pntwise = PointwiseIR::new(size.into())?;
    let min = pntwise.add_const(DValue::F32(min), 0);
    let max = pntwise.add_const(DValue::F32(max), 0);

    let w = pntwise.add_buf(TType::new(size, DType::F32));

    let old_w = pntwise.read(w, pntwise.tid(), 0)?;
    let high = pntwise.binary(old_w, max, CABinary::Min)?;
    let new_w = pntwise.binary(high, min, CABinary::Max)?;
    pntwise.write(w, pntwise.tid(), new_w)?;

    unsafe { pntwise.lower("clip".to_string()) }
}

#[derive(Clone, Debug)]
pub struct WeightClippingParams<T> {
    pub inner: T,
    pub placement: Placement,
    pub min: f32,
    pub max: f32,
}

impl<T: Default> Default for WeightClippingParams<T> {
    fn default() -> Self {
        Self { inner: T::default(), placement: Placement::After, min: -1.98, max: 1.98 }
    }
}

pub struct WeightClipping<G: Gpu, S: OptimiserState<G>> {
    inner: S,
    placement: Placement,
    op: CompiledKernel<G>,
    device: Arc<Device<G>>,
    size: usize,
}

impl<G: Gpu, S: OptimiserState<G>> OptimiserState<G> for WeightClipping<G, S> {
    type Params = WeightClippingParams<S::Params>;

    fn new(stream: &Arc<Stream<G>>, size: usize, params: Self::Params) -> Result<Self, G::Error> {
        let device = stream.device();

        Ok(Self {
            op: build_clip_op(size, params.min, params.max).unwrap().compile(device.clone())?,
            inner: S::new(stream, size, params.inner.clone())?,
            placement: params.placement,
            device,
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
        let src = build_clip_op(self.size, params.min, params.max).unwrap();
        self.op = src.compile(self.device.clone())?;
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
