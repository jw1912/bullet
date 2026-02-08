use std::{collections::HashMap, sync::Arc};

use bullet_compiler::{
    ir::{
        IRTrace,
        builder::IRBuilder,
        graph::{DType, TValue},
    },
    runtime::{Buffer, Device, ReadyToCompileGraph, Stream, TensorInput},
};

use super::{
    OptimiserState, OptimiserUpdateValue, WrapOptimiser,
    radam::{RAdam, RAdamParams},
    utils,
};

fn build_ranger_op(size: usize, alpha: f32) -> Result<ReadyToCompileGraph, IRTrace> {
    let builder = IRBuilder::default();

    let w = builder.add_input(size, DType::F32);
    let s = builder.add_input(size, DType::F32);

    let new_w = (((1.0 - alpha) * s)? + (alpha * w)?)?;
    let new_s = new_w.copy()?;

    let ir = builder.build([new_w, new_s]);

    ReadyToCompileGraph::new(
        ir,
        [
            ("w".to_string(), TensorInput::InOut(w.node(), new_w.node())),
            ("s".to_string(), TensorInput::InOut(s.node(), new_s.node())),
        ]
        .into(),
    )
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

pub struct RangerLookahead<D: Device, S> {
    inner: S,
    slow_params: Arc<D::Buffer>,
    k: usize,
    step: usize,
    op: D::CompiledGraph,
}

impl<D: Device, S: OptimiserState<D>> OptimiserState<D> for RangerLookahead<D, S> {
    type Params = RangerLookaheadParams<S::Params>;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::Error> {
        Ok(Self {
            op: device.compile(build_ranger_op(size, params.alpha).unwrap())?,
            slow_params: device.default_stream().make_blocking(&TValue::F32(vec![0.0; size]))?,
            inner: S::new(device, size, params.inner.clone())?,
            k: params.k,
            step: 0,
        })
    }

    fn update(
        &mut self,
        stream: &Arc<D::Stream>,
        weights: Arc<D::Buffer>,
        grads: Arc<D::Buffer>,
        gradient_factor: Arc<D::Buffer>,
        learning_rate: Arc<D::Buffer>,
    ) -> Result<OptimiserUpdateValue<D>, D::Error> {
        let mut blocks = Vec::new();

        self.step += 1;
        blocks.extend(self.inner.update(stream, weights.clone(), grads, gradient_factor, learning_rate)?);

        if self.step.is_multiple_of(self.k) {
            let args = [("w".into(), weights), ("s".into(), self.slow_params.clone())].into();
            blocks.push(stream.execute_graph(&self.op, &args)?);
        }

        Ok(blocks)
    }

    fn reset(&mut self) -> Result<(), D::Error> {
        self.inner.reset()?;
        self.step = 0;
        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), D::Error> {
        self.inner.set_params(params.inner)?;
        let device = self.slow_params.device();
        self.op = device.compile(build_ranger_op(self.slow_params.size(), params.alpha).unwrap())?;
        self.k = params.k;
        self.step = 0;
        Ok(())
    }

    fn load_from_checkpoint(map: &mut HashMap<String, &mut Self>, path: &str) -> Result<(), D::Error> {
        let slow_params = utils::load_weights_from_file(&format!("{path}/slow.bin"));

        for (id, par) in slow_params {
            let single = map.get_mut(&id).unwrap();
            let stream = single.slow_params.device().default_stream();
            stream.copy_h2d_blocking(&TValue::F32(par), single.slow_params.clone())?;
        }

        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.inner)).collect();
        S::load_from_checkpoint(&mut map, path)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::Error> {
        let stream = map.iter().next().unwrap().1.slow_params.device().default_stream();
        let slow_params: Vec<_> = map.iter().map(|(id, single)| (id, &single.slow_params)).collect();
        utils::write_weights_to_file::<D>(&stream, &slow_params, &format!("{path}/slow.bin"))?;

        let map = map.iter().map(|(id, single)| (id.clone(), &single.inner)).collect();
        S::write_to_checkpoint(&map, path)
    }
}

pub type Ranger<D> = WrapOptimiser<RangerLookahead<D, RAdam<D>>, RangerParams>;

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
