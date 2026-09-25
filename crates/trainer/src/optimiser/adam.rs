use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::tensor::{DType, IRTrace, TType, TValue};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseBuilder,
    runtime::{Device, DeviceProps, Gpu, Stream},
};

use crate::{
    model::{ModelDefinition, ModelWeights},
    optimiser::{Optimiser, OptimiserUpdateResult, OptimiserUpdateSync},
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
    pub fn build(&self, size: usize, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
        let (builder, p2size) = PointwiseBuilder::vectorised(size);

        let scalar = TType::new(1, DType::F32);
        let ty = TType::new(size, DType::F32);

        let adj_buf = builder.new_buffer(scalar);
        let rate_buf = builder.new_buffer(scalar);
        let grad_buf = builder.new_buffer(ty);
        let weight_buf = builder.new_buffer(ty);
        let momentum_buf = builder.new_buffer(ty);
        let velocity_buf = builder.new_buffer(ty);

        let adj = adj_buf.read(0, 0).splat(p2size);
        let rate = rate_buf.read(0, 0).splat(p2size);

        let tid = builder.tid();
        let grad = adj * grad_buf.read(tid, p2size);
        let weight = weight_buf.read(tid, p2size);
        let momentum = momentum_buf.read(tid, p2size);
        let velocity = velocity_buf.read(tid, p2size);

        let weight = weight * (1.0 - self.decay * rate);
        let momentum = self.beta1 * momentum + (1.0 - self.beta1) * grad;
        let velocity = self.beta2 * velocity + (1.0 - self.beta2) * grad * grad;

        let step = momentum * (velocity.sqrt() + 1e-8).recip();
        let weight = (weight - rate * step).max(self.min_weight).min(self.max_weight);

        weight_buf.write(tid, weight);
        momentum_buf.write(tid, momentum);
        velocity_buf.write(tid, velocity);

        builder.ir().eliminate_common_subexprs()?;

        unsafe { Ok(builder.inner().lower("adamw".to_string(), props)?) }
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

        let op = default_params.build(size, device.props()).unwrap().compile(device.clone())?;

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
        self.op = params.build(size, device.props()).unwrap().compile(device)?;
        Ok(())
    }
}
