use std::{collections::HashMap, sync::Arc};

use crate::{
    device::{Device, OperationError},
    graph::tensor::DenseMatrix,
    optimiser::utils,
};

use super::{
    radam::{RAdam, RAdamParams},
    OptimiserState, WrapOptimiser,
};

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
    slow_params: DenseMatrix<D>,
    alpha: f32,
    k: usize,
    step: usize,
}

impl<D: Device, S: OptimiserState<D>> OptimiserState<D> for RangerLookahead<D, S> {
    type Params = RangerLookaheadParams<S::Params>;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError> {
        Ok(Self {
            inner: S::new(device.clone(), size, params.inner.clone())?,
            slow_params: DenseMatrix::zeroed(device, size)?,
            alpha: params.alpha,
            k: params.k,
            step: 0,
        })
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>> {
        self.step += 1;
        self.inner.update(weights, grads, gradient_factor, learning_rate)?;

        if self.step % self.k == 0 {
            assert_eq!(weights.single_size, self.slow_params.single_size);
            assert!(self.slow_params.batch_size().is_none());

            self.slow_params.lerp(self.alpha, weights)?;
            weights.copy_from(&self.slow_params)?;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.inner.reset()?;
        self.step = 0;
        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) {
        self.inner.set_params(params.inner);
        self.alpha = params.alpha;
        self.k = params.k;
        self.step = 0;
    }

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
    ) -> Result<(), OperationError<D::DeviceError>> {
        let slow_params = utils::load_weights_from_file(&format!("{path}/slow.bin"), old_format);

        for (id, par) in &slow_params {
            let single = map.get_mut(id).unwrap();
            single.slow_params.load_from_slice(None, par)?;
        }

        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.inner)).collect();
        S::load_from_checkpoint(&mut map, path, old_format)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError> {
        let slow_params: Vec<_> = map.iter().map(|(id, single)| (id, &single.slow_params)).collect();
        utils::write_weights_to_file(&slow_params, &format!("{path}/slow.bin"))?;

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
