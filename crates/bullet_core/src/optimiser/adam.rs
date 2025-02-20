use std::{collections::HashMap, sync::Arc};

use crate::{
    device::{Device, OperationError},
    tensor::DenseMatrix,
};

use super::{
    clip::{WeightClipping, WeightClippingParams},
    decay::{WeightDecay, WeightDecayParams},
    utils::{self, Placement},
    OptimiserState, WrapOptimiser,
};

#[derive(Clone, Copy, Debug)]
pub struct AdamParams {
    pub beta1: f32,
    pub beta2: f32,
}

impl Default for AdamParams {
    fn default() -> Self {
        Self { beta1: 0.9, beta2: 0.999 }
    }
}

pub struct Adam<D: Device> {
    momentum: DenseMatrix<D>,
    velocity: DenseMatrix<D>,
    params: AdamParams,
}

impl<D: Device> OptimiserState<D> for Adam<D> {
    type Params = AdamParams;

    fn new(device: Arc<D>, size: usize, default_params: Self::Params) -> Result<Self, D::DeviceError> {
        Ok(Self {
            momentum: DenseMatrix::zeroed(device.clone(), size)?,
            velocity: DenseMatrix::zeroed(device, size)?,
            params: default_params,
        })
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>> {
        assert!(weights.batch_size().is_none());
        assert!(self.momentum.batch_size().is_none());
        assert!(self.velocity.batch_size().is_none());
        assert_eq!(weights.size(), self.momentum.size());
        assert_eq!(weights.size(), self.velocity.size());

        D::adam(
            weights.size(),
            &mut weights.buf,
            &grads.buf,
            &mut self.momentum.buf,
            &mut self.velocity.buf,
            self.params.beta1,
            self.params.beta2,
            gradient_factor,
            learning_rate,
            true,
        )
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.momentum.set_zero()?;
        self.velocity.set_zero()
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError> {
        let momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.momentum)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        utils::write_weights_to_file(&momentum, &format!("{path}/momentum.bin"))?;
        utils::write_weights_to_file(&velocity, &format!("{path}/velocity.bin"))
    }

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
    ) -> Result<(), D::DeviceError> {
        let paths = [format!("{path}/momentum.bin"), format!("{path}/velocity.bin")];
        let mut momentum = utils::load_weights_from_file(&paths[0], old_format);
        let mut velocity = utils::load_weights_from_file(&paths[1], old_format);

        momentum.sort_by_key(|(id, _)| id.clone());
        velocity.sort_by_key(|(id, _)| id.clone());

        for ((id1, mom), (id2, vel)) in momentum.iter().zip(velocity.iter()) {
            assert_eq!(id1, id2);

            let single = map.get_mut(id1).unwrap();
            single.momentum.load_from_slice(None, mom)?;
            single.velocity.load_from_slice(None, vel)?;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) {
        self.params = params;
    }
}

type AdamWClip<D> = WeightClipping<WeightDecay<Adam<D>>>;
pub type AdamW<D> = WrapOptimiser<AdamWClip<D>, AdamWParams>;

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
