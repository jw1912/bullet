use std::{collections::HashMap, sync::Arc};

use crate::{
    device::{Device, OperationError},
    graph::tensor::DenseMatrix,
};

use super::{utils::Placement, OptimiserState};

#[derive(Clone, Debug)]
pub struct WeightClippingParams<T> {
    pub inner: T,
    pub placement: Placement,
    pub min: f32,
    pub max: f32,
}

impl<T: Default> Default for WeightClippingParams<T> {
    fn default() -> Self {
        Self { inner: T::default(), placement: Placement::Before, min: -1.98, max: 1.98 }
    }
}

pub struct WeightClipping<S> {
    inner: S,
    placement: Placement,
    min: f32,
    max: f32,
}

impl<D: Device, S: OptimiserState<D>> OptimiserState<D> for WeightClipping<S> {
    type Params = WeightClippingParams<S::Params>;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError> {
        Ok(Self {
            inner: S::new(device, size, params.inner.clone())?,
            placement: params.placement,
            min: params.min,
            max: params.max,
        })
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>> {
        if self.placement == Placement::Before {
            weights.clamp(self.min, self.max)?;
        }

        self.inner.update(weights, grads, gradient_factor, learning_rate)?;

        if self.placement == Placement::After {
            weights.clamp(self.min, self.max)?;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.inner.reset()
    }

    fn set_params(&mut self, params: Self::Params) {
        self.inner.set_params(params.inner);
        self.min = params.min;
        self.max = params.max;
    }

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
    ) -> Result<(), OperationError<D::DeviceError>> {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.inner)).collect();
        S::load_from_checkpoint(&mut map, path, old_format)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError> {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.inner)).collect();
        S::write_to_checkpoint(&map, path)
    }
}
