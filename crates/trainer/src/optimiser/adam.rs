use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::tensor::{DType, IRTrace, TType};
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

use super::{OptimiserState, bf16, utils};

#[derive(Clone, Copy, Debug)]
pub struct AdamWParams {
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub min_weight: f32,
    pub max_weight: f32,
    /// Store momentum and velocity as bf16, see [`bf16`].
    pub bf16_state: bool,
}

impl Default for AdamWParams {
    fn default() -> Self {
        Self { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98, bf16_state: false }
    }
}

impl AdamWParams {
    pub fn build(&self, size: usize, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
        let (builder, p2size) = PointwiseBuilder::vectorised(size);

        let scalar = TType::new(1, DType::F32);
        let ty = TType::new(size, DType::F32);
        let state_ty = bf16::state_type(size, self.bf16_state);

        let adj_buf = builder.new_buffer(scalar);
        let rate_buf = builder.new_buffer(scalar);
        let seed_bufs = self.bf16_state.then(|| bf16::seed_buffers(&builder, p2size));
        let grad_buf = builder.new_buffer(ty);
        let weight_buf = builder.new_buffer(ty);
        let momentum_buf = builder.new_buffer(state_ty);
        let velocity_buf = builder.new_buffer(state_ty);

        let adj = adj_buf.read(0, 0).splat(p2size);
        let rate = rate_buf.read(0, 0).splat(p2size);

        let tid = builder.tid();
        let grad = adj * grad_buf.read(tid, p2size);
        let weight = weight_buf.read(tid, p2size);
        let momentum = bf16::to_f32(momentum_buf.read(tid, p2size));
        let velocity = bf16::to_f32(velocity_buf.read(tid, p2size));

        let weight = weight * (1.0 - self.decay * rate);
        let momentum = self.beta1 * momentum + (1.0 - self.beta1) * grad;
        let velocity = self.beta2 * velocity + (1.0 - self.beta2) * grad * grad;

        let step = momentum * (velocity.sqrt() + 1e-8).recip();
        let weight = (weight - rate * step).max(self.min_weight).min(self.max_weight);

        weight_buf.write(tid, weight);
        if let Some((seed, lanes)) = seed_bufs {
            momentum_buf.write(tid, momentum.cast(DType::Bf16));
            velocity_buf.write(tid, bf16::round_stochastic(velocity, bf16::uniform(&builder, seed, lanes, p2size)));
        } else {
            momentum_buf.write(tid, momentum);
            velocity_buf.write(tid, velocity);
        }

        builder.ir().eliminate_common_subexprs()?;

        unsafe { Ok(builder.inner().lower("adamw".to_string(), props)?) }
    }
}

pub struct AdamW<G: Gpu> {
    momentum: Arc<Buffer<G>>,
    velocity: Arc<Buffer<G>>,
    op: CompiledKernel<G>,
    seed: Option<bf16::Seed<G>>,
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

    /// The velocity as f32 (tests compare bf16 and f32 state).
    #[cfg(all(test, any(feature = "cuda", feature = "rocm", feature = "metal")))]
    pub(crate) fn velocity_f32(&self) -> Result<Vec<f32>, G::Error> {
        let bullet_compiler::tensor::TValue::F32(v) = bf16::as_f32(&self.velocity)?.to_host()? else {
            unreachable!("f32 copy")
        };
        Ok(v)
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

        let zeros = bf16::zeros(size, default_params.bf16_state);

        Ok(Self {
            momentum: Buffer::from_host(device, &zeros)?,
            velocity: Buffer::from_host(device, &zeros)?,
            op,
            seed: if default_params.bf16_state { Some(bf16::Seed::new(device, size)?) } else { None },
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

        let mut inputs = vec![gradient_factor, learning_rate];
        if let Some(seed) = &mut self.seed {
            inputs.extend(seed.inputs());
            sync.push_copy(seed.advance(stream)?);
        }
        inputs.push(grads);

        sync.push_kernel(self.op.execute(
            stream.clone(),
            inputs,
            vec![weights, self.momentum.clone(), self.velocity.clone()],
        )?);

        Ok(sync)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        let zeros = bf16::zeros(self.momentum.size(), self.seed.is_some());
        self.momentum.copy_from_host(&zeros)?;
        self.velocity.copy_from_host(&zeros)?;
        Ok(())
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        // Checkpoints hold f32 whatever the storage, so they move between runs.
        let momentum =
            map.iter().map(|(id, s)| Ok((id, bf16::as_f32(&s.momentum)?))).collect::<Result<Vec<_>, G::Error>>()?;
        let velocity =
            map.iter().map(|(id, s)| Ok((id, bf16::as_f32(&s.velocity)?))).collect::<Result<Vec<_>, G::Error>>()?;
        let momentum: Vec<_> = momentum.iter().map(|(id, b)| (*id, b)).collect();
        let velocity: Vec<_> = velocity.iter().map(|(id, b)| (*id, b)).collect();
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
            let bf = single.seed.is_some();
            single.momentum.copy_from_host(&bf16::from_f32(mom, bf))?;
            single.velocity.copy_from_host(&bf16::from_f32(vel, bf))?;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        // Switching the state storage keeps the current moments.
        if self.seed.is_some() != params.bf16_state {
            self.momentum = bf16::convert(&self.momentum, params.bf16_state)?;
            self.velocity = bf16::convert(&self.velocity, params.bf16_state)?;
            self.seed = if params.bf16_state {
                Some(bf16::Seed::new(&self.momentum.device(), self.momentum.size())?)
            } else {
                None
            };
        }
        let size = self.momentum.size();
        let device = self.momentum.device();
        self.op = params.build(size, device.props()).unwrap().compile(device)?;
        Ok(())
    }
}
