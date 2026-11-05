use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
};

use bullet_compiler::tensor::{DType, DValue, IRTrace, TType, TValue};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseBuilder,
    runtime::{Device, DeviceProps, Gpu, Stream},
};

use crate::optimiser::{OptimiserUpdateResult, OptimiserUpdateSync};

use super::{OptimiserState, bf16, utils};

#[derive(Clone, Copy, Debug)]
pub struct RAdamParams {
    pub beta1: f32,
    pub beta2: f32,
    pub n_sma_threshold: f32,
    pub decay: f32,
    pub clip: Option<(f32, f32)>,
    /// Store momentum and velocity as bf16, see [`bf16`].
    pub bf16_state: bool,
}

impl Default for RAdamParams {
    fn default() -> Self {
        Self { beta1: 0.9, beta2: 0.999, n_sma_threshold: 5.0, decay: 0.0, clip: None, bf16_state: false }
    }
}

impl RAdamParams {
    pub fn build(&self, size: usize, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
        let (min, max) = self.clip.unwrap_or((f32::MIN, f32::MAX));

        let (builder, p2size) = PointwiseBuilder::vectorised(size);

        let scalar = TType::new(1, DType::F32);
        let ty = TType::new(size, DType::F32);
        let state_ty = bf16::state_type(size, self.bf16_state);

        let adj_buf = builder.new_buffer(scalar);
        let rate_buf = builder.new_buffer(scalar);
        let step_size_buf = builder.new_buffer(scalar);
        let denom_buf = builder.new_buffer(TType::new(1, DType::I32));
        let seed_bufs = self.bf16_state.then(|| bf16::seed_buffers(&builder, p2size));
        let grad_buf = builder.new_buffer(ty);
        let weight_buf = builder.new_buffer(ty);
        let momentum_buf = builder.new_buffer(state_ty);
        let velocity_buf = builder.new_buffer(state_ty);

        let adj = adj_buf.read(0, 0).splat(p2size);
        let rate = (rate_buf.read(0, 0) * step_size_buf.read(0, 0)).splat(p2size);
        let denom = denom_buf.read(0, 0).cast(DType::F32).splat(p2size);

        let tid = builder.tid();
        let grad = adj * grad_buf.read(tid, p2size);
        let weight = weight_buf.read(tid, p2size);
        let momentum = bf16::to_f32(momentum_buf.read(tid, p2size));
        let velocity = bf16::to_f32(velocity_buf.read(tid, p2size));

        let weight = weight * (1.0 - self.decay * rate);
        let momentum = self.beta1 * momentum + (1.0 - self.beta1) * grad;
        let velocity = self.beta2 * velocity + (1.0 - self.beta2) * grad * grad;

        // `denom` is written as 0 or 1, and picks out whether the step is scaled
        // by `1 / (sqrt(v) + eps)` - as a lerp, since there is no select op
        let scale = 1.0 - denom + denom * (velocity.sqrt() + 1e-8).recip();
        let step = momentum * scale;
        let weight = (weight - rate * step).max(min).min(max);

        weight_buf.write(tid, weight);
        if let Some((seed, lanes)) = seed_bufs {
            momentum_buf.write(tid, momentum.cast(DType::Bf16));
            velocity_buf.write(tid, bf16::round_stochastic(velocity, bf16::uniform(&builder, seed, lanes, p2size)));
        } else {
            momentum_buf.write(tid, momentum);
            velocity_buf.write(tid, velocity);
        }

        builder.ir().eliminate_common_subexprs()?;

        unsafe { Ok(builder.inner().lower("radam".to_string(), props)?) }
    }
}

pub struct RAdam<G: Gpu> {
    momentum: Arc<Buffer<G>>,
    velocity: Arc<Buffer<G>>,
    op: CompiledKernel<G>,
    params: RAdamParams,
    step: usize,
    step_size: Arc<Buffer<G>>,
    denom: Arc<Buffer<G>>,
    cpu_step_size: TValue,
    cpu_denom: TValue,
    seed: Option<bf16::Seed<G>>,
}

impl<G: Gpu> RAdam<G> {
    /// The velocity as f32 (tests compare bf16 and f32 state).
    #[cfg(all(test, any(feature = "cuda", feature = "rocm", feature = "metal")))]
    pub(crate) fn velocity_f32(&self) -> Result<Vec<f32>, G::Error> {
        let TValue::F32(v) = bf16::as_f32(&self.velocity)?.to_host()? else { unreachable!("f32 copy") };
        Ok(v)
    }
}

impl<G: Gpu> OptimiserState<G> for RAdam<G> {
    type Params = RAdamParams;

    fn new(device: &Arc<Device<G>>, size: usize, default_params: Self::Params) -> Result<Self, G::Error> {
        let op = default_params.build(size, device.props()).unwrap().compile(device.clone())?;

        let zeros = bf16::zeros(size, default_params.bf16_state);

        Ok(Self {
            momentum: Buffer::from_host(device, &zeros)?,
            velocity: Buffer::from_host(device, &zeros)?,
            op,
            params: default_params,
            step: 0,
            step_size: Buffer::from_host(device, &TValue::zeros(DType::F32, 1))?,
            denom: Buffer::from_host(device, &TValue::zeros(DType::I32, 1))?,
            cpu_step_size: TValue::F32(vec![0.0]),
            cpu_denom: TValue::I32(vec![0]),
            seed: if default_params.bf16_state { Some(bf16::Seed::new(device, size)?) } else { None },
        })
    }

    #[allow(unused)]
    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
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

        let denom = i32::from(n_sma > params.n_sma_threshold);

        self.cpu_step_size.write(0, DValue::F32(step_size));
        self.cpu_denom.write(0, DValue::I32(denom));

        let mut sync = OptimiserUpdateSync::default();

        sync.push_copy(self.step_size.copy_from_host_async(stream, &self.cpu_step_size)?);
        sync.push_copy(self.denom.copy_from_host_async(stream, &self.cpu_denom)?);

        let mut inputs = vec![gradient_factor, learning_rate, self.step_size.clone(), self.denom.clone()];
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
        self.step = 0;
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
        utils::write_weights_to_file::<G>(&velocity, &format!("{path}/velocity.bin"))?;

        let mut file = File::create(format!("{path}/step.txt")).unwrap();
        for (id, single) in map.iter() {
            writeln!(file, "{id},{}", single.step).unwrap();
        }

        Ok(())
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let paths = [format!("{path}/momentum.bin"), format!("{path}/velocity.bin")];
        let mut momentum = utils::load_weights_from_file(&paths[0]);
        let mut velocity = utils::load_weights_from_file(&paths[1]);

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

        for (((id1, mom), (id2, vel)), (id3, step)) in momentum.into_iter().zip(velocity).zip(steps) {
            assert_eq!(id1, id2);
            assert_eq!(id1, id3);

            let single = map.get_mut(&id1).unwrap();
            let bf = single.seed.is_some();
            single.momentum.copy_from_host(&bf16::from_f32(mom, bf))?;
            single.velocity.copy_from_host(&bf16::from_f32(vel, bf))?;
            single.step = step;
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
        self.params = params;

        let size = self.momentum.size();
        let device = self.momentum.device();
        self.op = params.build(size, device.props()).unwrap().compile(device)?;
        Ok(())
    }
}
