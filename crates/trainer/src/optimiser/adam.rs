use std::{collections::BTreeMap, io, sync::Arc};

use bullet_compiler::tensor::{DType, TValue};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    optimiser::build_adamw_op,
    runtime::{Device, DeviceProps, Gpu, Stream},
};

use crate::{
    model::{ModelDefinition, ModelWeights},
    optimiser::{CpuOptimiserState, Optimiser, OptimiserUpdateResult, OptimiserUpdateSync},
};

use super::{OptimiserState, utils};

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

impl AdamWParams {
    pub fn build(&self, size: usize, props: &DeviceProps) -> KernelSrc {
        build_adamw_op(size, props.dialect(), self.decay, self.beta1, self.beta2, self.min_weight, self.max_weight)
    }
}

pub struct AdamW<G: Gpu> {
    momentum: Arc<Buffer<G>>,
    velocity: Arc<Buffer<G>>,
    op: CompiledKernel<G>,
}

impl<G: Gpu> AdamW<G> {
    pub fn new(
        definition: ModelDefinition,
        weights: ModelWeights,
        device: Arc<Device<G>>,
        params: AdamWParams,
    ) -> Result<Optimiser<G, Self>, G::Error> {
        Optimiser::new(definition, weights, device, params)
    }
}

impl<G: Gpu> OptimiserState<G> for AdamW<G> {
    type Params = AdamWParams;

    fn new(device: &Arc<Device<G>>, size: usize, default_params: Self::Params) -> Result<Self, G::Error> {
        if default_params.max_weight < default_params.min_weight {
            return Err(
                format!("Invalid clipping: {} >= {}", default_params.min_weight, default_params.max_weight).into()
            );
        }

        let op = default_params.build(size, device.props()).compile(device.clone())?;

        Ok(Self {
            momentum: Buffer::from_host(device, &TValue::zeros(DType::F32, size))?,
            velocity: Buffer::from_host(device, &TValue::zeros(DType::F32, size))?,
            op,
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
        let mut sync = OptimiserUpdateSync::default();

        sync.push_kernel(self.op.execute(
            stream.clone(),
            vec![gradient_factor, learning_rate, grads],
            vec![weights, self.momentum.clone(), self.velocity.clone()],
        )?);

        Ok(sync)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        let size = self.momentum.size();
        self.momentum.copy_from_host(&TValue::zeros(DType::F32, size))?;
        self.velocity.copy_from_host(&TValue::zeros(DType::F32, size))?;
        Ok(())
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.momentum)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        utils::write_weights_to_file::<G>(&momentum, &format!("{path}/momentum.bin"))?;
        utils::write_weights_to_file::<G>(&velocity, &format!("{path}/velocity.bin"))
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let paths = [format!("{path}/momentum.bin"), format!("{path}/velocity.bin")];
        let mut momentum = utils::load_weights_from_file(&paths[0]);
        let mut velocity = utils::load_weights_from_file(&paths[1]);

        momentum.sort_by_key(|(id, _)| id.clone());
        velocity.sort_by_key(|(id, _)| id.clone());

        for ((id1, mom), (id2, vel)) in momentum.into_iter().zip(velocity) {
            assert_eq!(id1, id2);

            let single = map.get_mut(&id1).unwrap();
            single.momentum.copy_from_host(&TValue::F32(mom))?;
            single.velocity.copy_from_host(&TValue::F32(vel))?;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        let size = self.momentum.size();
        let device = self.momentum.device();
        self.op = params.build(size, device.props()).compile(device)?;
        Ok(())
    }
}

pub struct CpuAdamW {
    momentum: Vec<f32>,
    velocity: Vec<f32>,
    params: AdamWParams,
}

impl CpuOptimiserState for CpuAdamW {
    type Params = AdamWParams;

    fn new(size: usize, default_params: Self::Params) -> Self {
        assert!(default_params.max_weight >= default_params.min_weight);

        Self { momentum: vec![0.0; size], velocity: vec![0.0; size], params: default_params }
    }

    fn update(&mut self, weights: &mut TValue, grads: &mut TValue, gradient_factor: f32, learning_rate: f32) {
        let TValue::F32(weights) = weights else { panic!() };
        let TValue::F32(grads) = grads else { panic!() };

        let size = self.momentum.len();

        assert_eq!(size, weights.len());
        assert_eq!(size, grads.len());
        assert_eq!(size, self.velocity.len());

        let AdamWParams { decay, beta1, beta2, min_weight, max_weight } = self.params;

        for i in 0..size {
            weights[i] *= 1.0 - decay * learning_rate;
            let grad = gradient_factor * grads[i];

            self.momentum[i] = beta1 * self.momentum[i] + (1.0 - beta1) * grad;
            self.velocity[i] = beta2 * self.velocity[i] + (1.0 - beta2) * grad * grad;

            let val = self.momentum[i] / (self.velocity[i].sqrt() + 0.00000001);
            weights[i] -= learning_rate * val;
            weights[i] = weights[i].max(min_weight).min(max_weight);
        }
    }

    fn reset(&mut self) {
        let size = self.momentum.len();
        self.momentum = vec![0.0; size];
        self.velocity = vec![0.0; size];
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> io::Result<()> {
        let momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.momentum)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        utils::write_mapped_weights_to_file(&momentum, &format!("{path}/momentum.bin"), |x| x.clone())?;
        utils::write_mapped_weights_to_file(&velocity, &format!("{path}/velocity.bin"), |x| x.clone())
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> io::Result<()> {
        let paths = [format!("{path}/momentum.bin"), format!("{path}/velocity.bin")];
        let mut momentum = utils::load_weights_from_file(&paths[0]);
        let mut velocity = utils::load_weights_from_file(&paths[1]);

        momentum.sort_by_key(|(id, _)| id.clone());
        velocity.sort_by_key(|(id, _)| id.clone());

        for ((id1, mom), (id2, vel)) in momentum.into_iter().zip(velocity) {
            assert_eq!(id1, id2);

            let single = map.get_mut(&id1).unwrap();
            single.momentum = mom;
            single.velocity = vel;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) {
        self.params = params
    }
}
