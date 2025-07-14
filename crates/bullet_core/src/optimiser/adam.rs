use std::{collections::HashMap, sync::Arc};

use crate::{
    device::{
        base::{AdamConfig, BaseOperations},
        Device, OperationError,
    },
    graph::tensor::DenseMatrix,
};

use super::{utils, OptimiserState};

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

pub struct AdamW<D: Device> {
    momentum: DenseMatrix<D>,
    velocity: DenseMatrix<D>,
    params: AdamWParams,
}

impl<D: Device> OptimiserState<D> for AdamW<D> {
    type Params = AdamWParams;

    fn new(device: Arc<D>, size: usize, default_params: Self::Params) -> Result<Self, D::DeviceError> {
        if default_params.max_weight < default_params.min_weight {
            return Err(D::DeviceError::default());
        }

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

        let (min, max) = (self.params.min_weight, self.params.max_weight);
        let clip = (min != max).then_some((min, max));

        let cfg = AdamConfig {
            beta1: self.params.beta1,
            beta2: self.params.beta2,
            gradient_factor,
            learning_rate,
            denom: true,
            decay: 1.0 - self.params.decay * learning_rate,
            clip,
        };

        weights.buf.adam(&cfg, weights.size(), &grads.buf, &mut self.momentum.buf, &mut self.velocity.buf)?;

        Ok(())
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.momentum.set_to(0.0)?;
        self.velocity.set_to(0.0)
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
    ) -> Result<(), OperationError<D::DeviceError>> {
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
