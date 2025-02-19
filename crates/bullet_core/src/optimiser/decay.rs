use std::{collections::HashMap, sync::Arc};

use crate::{device::Device, tensor::DenseMatrix};

use super::{utils::Placement, OptimiserState};

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

pub struct WeightDecay<S> {
    inner: S,
    placement: Placement,
    decay: f32,
}

impl<D: Device, S: OptimiserState<D>> OptimiserState<D> for WeightDecay<S> {
    type Params = WeightDecayParams<S::Params>;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Self {
        Self { inner: S::new(device, size, params.inner.clone()), placement: params.placement, decay: params.decay }
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) {
        let factor = 1.0 - self.decay * learning_rate;

        if self.placement == Placement::Before {
            D::linear_comb_single(weights.size(), factor, None, 0.0, None, &mut weights.buf);
        }

        self.inner.update(weights, grads, gradient_factor, learning_rate);

        if self.placement == Placement::After {
            D::linear_comb_single(weights.size(), factor, None, 0.0, None, &mut weights.buf);
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn set_params(&mut self, params: Self::Params) {
        self.inner.set_params(params.inner);
        self.decay = params.decay;
    }

    fn load_from_checkpoint(map: &mut HashMap<String, &mut Self>, path: &str, old_format: bool) {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.inner)).collect();
        S::load_from_checkpoint(&mut map, path, old_format);
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.inner)).collect();
        S::write_to_checkpoint(&map, path);
    }
}
