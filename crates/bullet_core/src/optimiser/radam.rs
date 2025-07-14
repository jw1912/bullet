use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
};

use crate::{
    device::{
        base::{AdamConfig, BaseOperations},
        Device, OperationError,
    },
    graph::tensor::DenseMatrix,
};

use super::{utils, OptimiserState};

#[derive(Clone, Copy, Debug)]
pub struct RAdamParams {
    pub beta1: f32,
    pub beta2: f32,
    pub n_sma_threshold: f32,
    pub decay: f32,
    pub clip: Option<(f32, f32)>,
}

impl Default for RAdamParams {
    fn default() -> Self {
        Self { beta1: 0.9, beta2: 0.999, n_sma_threshold: 5.0, decay: 0.0, clip: None }
    }
}

pub struct RAdam<D: Device> {
    momentum: DenseMatrix<D>,
    velocity: DenseMatrix<D>,
    params: RAdamParams,
    step: usize,
}

impl<D: Device> OptimiserState<D> for RAdam<D> {
    type Params = RAdamParams;

    fn new(device: Arc<D>, size: usize, default_params: Self::Params) -> Result<Self, D::DeviceError> {
        Ok(Self {
            momentum: DenseMatrix::zeroed(device.clone(), size)?,
            velocity: DenseMatrix::zeroed(device, size)?,
            params: default_params,
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
        assert!(weights.batch_size().is_none());
        assert!(self.momentum.batch_size().is_none());
        assert!(self.velocity.batch_size().is_none());
        assert_eq!(weights.size(), self.momentum.size());
        assert_eq!(weights.size(), self.velocity.size());

        self.step += 1;

        let params = self.params;
        let step = self.step as f32;

        let beta2_t = params.beta2.powf(step);
        let n_sma_max = 2.0 / (1.0 - params.beta2) - 1.0;
        let n_sma = n_sma_max - 2.0 * step * beta2_t / (1.0 - beta2_t);

        let denom = 1.0 - params.beta1.powf(step);
        let step_size = if n_sma > params.n_sma_threshold {
            let p1 = (n_sma - 4.0) / (n_sma_max - 4.0);
            let p2 = (n_sma - 2.0) / n_sma;
            let p3 = n_sma_max / (n_sma_max - 2.0);
            ((1.0 - beta2_t) * p1 * p2 * p3).sqrt() / denom
        } else {
            1.0 / denom
        };

        let lr = learning_rate * step_size;

        let cfg = AdamConfig {
            beta1: self.params.beta1,
            beta2: self.params.beta2,
            gradient_factor,
            learning_rate: lr,
            denom: n_sma > params.n_sma_threshold,
            clip: self.params.clip,
            decay: 1.0 - self.params.decay * lr,
        };

        weights.buf.adam(&cfg, weights.size(), &grads.buf, &mut self.momentum.buf, &mut self.velocity.buf)?;

        Ok(())
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.step = 0;
        self.momentum.set_to(0.0)?;
        self.velocity.set_to(0.0)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError> {
        let momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.momentum)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        utils::write_weights_to_file(&momentum, &format!("{path}/momentum.bin"))?;
        utils::write_weights_to_file(&velocity, &format!("{path}/velocity.bin"))?;

        let mut file = File::create(format!("{path}/step.txt")).unwrap();
        for (id, single) in map.iter() {
            writeln!(file, "{id},{}", single.step).unwrap();
        }

        Ok(())
    }

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
    ) -> Result<(), OperationError<D::DeviceError>> {
        let paths = [format!("{path}/momentum.bin"), format!("{path}/velocity.bin")];
        let mut momentum = utils::load_weights_from_file(&paths[0], old_format);
        let mut velocity = utils::load_weights_from_file(&paths[1], old_format);

        let file = File::open(format!("{path}/step.txt")).unwrap();
        let mut steps = BufReader::new(file)
            .lines()
            .map(|s| {
                let s = s.unwrap();
                let mut split = s.split(',');
                let id = split.next().unwrap();
                (id.to_string(), split.next().unwrap().parse().unwrap())
            })
            .collect::<Vec<(String, usize)>>();

        momentum.sort_by_key(|(id, _)| id.clone());
        velocity.sort_by_key(|(id, _)| id.clone());
        steps.sort_by_key(|(id, _)| id.clone());

        for (((id1, mom), (id2, vel)), (id3, step)) in momentum.iter().zip(velocity.iter()).zip(steps.iter()) {
            assert_eq!(id1, id2);
            assert_eq!(id1, id3);

            let single = map.get_mut(id1).unwrap();
            single.momentum.load_from_slice(None, mom)?;
            single.velocity.load_from_slice(None, vel)?;
            single.step = *step;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) {
        self.params = params;
    }
}
