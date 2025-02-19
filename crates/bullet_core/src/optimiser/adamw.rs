use std::{collections::HashMap, sync::Arc};

use crate::{device::Device, tensor::DenseMatrix};

use super::{
    adam::{Adam, AdamParams},
    clip::{WeightClipping, WeightClippingParams},
    decay::{WeightDecay, WeightDecayParams},
    utils::Placement,
    OptimiserState,
};

pub type AdamWClip<D> = WeightClipping<WeightDecay<Adam<D>>>;

// The below code exists for backwards-compatibility

#[derive(Clone, Copy, Debug)]
pub struct AdamWParams {
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for AdamWParams {
    fn default() -> Self {
        Self { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 }
    }
}

impl From<AdamWParams> for WeightClippingParams<WeightDecayParams<AdamParams>> {
    fn from(value: AdamWParams) -> Self {
        WeightClippingParams {
            inner: WeightDecayParams {
                inner: AdamParams { beta1: value.beta1, beta2: value.beta2 },
                decay: value.decay,
                placement: Placement::Before,
            },
            min: value.min_weight,
            max: value.max_weight,
            placement: Placement::After,
        }
    }
}

pub struct AdamW<D: Device> {
    inner: AdamWClip<D>,
}

impl<D: Device> OptimiserState<D> for AdamW<D> {
    type Params = AdamWParams;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Self {
        Self { inner: AdamWClip::new(device, size, params.into()) }
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) {
        self.inner.update(weights, grads, gradient_factor, learning_rate);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn set_params(&mut self, params: Self::Params) {
        self.inner.set_params(params.into());
    }

    fn load_from_checkpoint(map: &mut HashMap<String, &mut Self>, path: &str, old_format: bool) {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.inner)).collect();
        AdamWClip::<D>::load_from_checkpoint(&mut map, path, old_format);
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.inner)).collect();
        AdamWClip::<D>::write_to_checkpoint(&map, path);
    }
}
