use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::{
    ir::IRError,
    tensor::{DType, TType, TValue, operation::CABinary},
};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseIR,
    runtime::{Device, Gpu, Stream},
};

use crate::optimiser::{OptimiserUpdateResult, OptimiserUpdateSync};

use super::{
    OptimiserState, WrapOptimiser,
    radam::{RAdam, RAdamParams},
    utils,
};

fn build_ranger_op(size: usize, alpha: f32) -> Result<KernelSrc, IRError> {
    let mut pntwise = PointwiseIR::new(size.into())?;

    let w = pntwise.add_buf(TType::new(size, DType::F32));
    let s = pntwise.add_buf(TType::new(size, DType::F32));
    let old_w = pntwise.read(w, pntwise.tid(), 0)?;
    let old_s = pntwise.read(s, pntwise.tid(), 0)?;

    let wweight = pntwise.add_const(alpha.into(), 0);
    let lhs = pntwise.binary(wweight, old_w, CABinary::Mul)?;

    let sweight = pntwise.add_const((1.0 - alpha).into(), 0);
    let rhs = pntwise.binary(sweight, old_s, CABinary::Mul)?;

    let new_w = pntwise.binary(lhs, rhs, CABinary::Add)?;
    pntwise.write(w, pntwise.tid(), new_w)?;
    pntwise.write(s, pntwise.tid(), new_w)?;

    unsafe { pntwise.lower("ranger".to_string()) }
}

#[derive(Clone, Debug)]
pub struct RangerLookaheadParams<T> {
    pub inner: T,
    pub alpha: f32,
    pub k: usize,
}

impl<T: Default> Default for RangerLookaheadParams<T> {
    fn default() -> Self {
        Self { inner: T::default(), alpha: 0.5, k: 6 }
    }
}

pub struct RangerLookahead<G: Gpu, S> {
    inner: S,
    slow_params: Arc<Buffer<G>>,
    k: usize,
    step: usize,
    op: CompiledKernel<G>,
}

impl<G: Gpu, S: OptimiserState<G>> OptimiserState<G> for RangerLookahead<G, S> {
    type Params = RangerLookaheadParams<S::Params>;

    fn new(device: &Arc<Device<G>>, size: usize, params: Self::Params) -> Result<Self, G::Error> {
        Ok(Self {
            op: build_ranger_op(size, params.alpha).unwrap().compile(device.clone())?,
            slow_params: Buffer::from_host(device, &TValue::F32(vec![0.0; size]))?,
            inner: S::new(device, size, params.inner.clone())?,
            k: params.k,
            step: 0,
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

        self.step += 1;
        blocks.extend_by(self.inner.update(stream, weights.clone(), grads, gradient_factor, learning_rate)?);

        if self.step.is_multiple_of(self.k) {
            blocks.push_kernel(self.op.execute(stream.clone(), Vec::new(), vec![weights, self.slow_params.clone()])?);
        }

        Ok(blocks)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        self.inner.reset()?;
        self.step = 0;
        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        self.inner.set_params(params.inner)?;
        let device = self.slow_params.device();
        self.op = build_ranger_op(self.slow_params.size(), params.alpha).unwrap().compile(device)?;
        self.k = params.k;
        self.step = 0;
        Ok(())
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let slow_params = utils::load_weights_from_file(&format!("{path}/slow.bin"));

        for (id, par) in slow_params {
            let single = map.get_mut(&id).unwrap();
            single.slow_params.copy_from_host(&TValue::F32(par))?;
        }

        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.inner)).collect();
        S::load_from_checkpoint(&mut map, path)
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let slow_params: Vec<_> = map.iter().map(|(id, single)| (id, &single.slow_params)).collect();
        utils::write_weights_to_file::<G>(&slow_params, &format!("{path}/slow.bin"))?;

        let map = map.iter().map(|(id, single)| (id.clone(), &single.inner)).collect();
        S::write_to_checkpoint(&map, path)
    }
}

pub type Ranger<G> = WrapOptimiser<RangerLookahead<G, RAdam<G>>, RangerParams>;

#[derive(Clone, Copy, Debug)]
pub struct RangerParams {
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub min_weight: f32,
    pub max_weight: f32,
    pub alpha: f32,
    pub k: usize,
}

impl Default for RangerParams {
    fn default() -> Self {
        RangerParams { decay: 0.01, beta1: 0.99, beta2: 0.999, min_weight: -1.98, max_weight: 1.98, alpha: 0.5, k: 6 }
    }
}

impl From<RangerParams> for RangerLookaheadParams<RAdamParams> {
    fn from(value: RangerParams) -> Self {
        RangerLookaheadParams {
            inner: RAdamParams {
                beta1: value.beta1,
                beta2: value.beta2,
                n_sma_threshold: 5.0,
                decay: value.decay,
                clip: Some((value.min_weight, value.max_weight)),
            },
            alpha: value.alpha,
            k: value.k,
        }
    }
}
