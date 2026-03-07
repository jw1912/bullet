use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
};

use bullet_compiler::tensor::{DType, DValue, IRTrace, TValue};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    runtime::{Gpu, Stream},
};

use crate::optimiser::OptimiserUpdateResult;

use super::{OptimiserState, utils};

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

impl RAdamParams {
    pub fn build(&self, _size: usize) -> Result<KernelSrc, IRTrace> {
        unimplemented!()
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
}

impl<G: Gpu> OptimiserState<G> for RAdam<G> {
    type Params = RAdamParams;

    fn new(stream: &Arc<Stream<G>>, size: usize, default_params: Self::Params) -> Result<Self, G::Error> {
        let op = default_params.build(size).unwrap().compile(stream.device())?;

        Ok(Self {
            momentum: Buffer::from_host(stream, &TValue::zeros(DType::F32, size))?.value().0,
            velocity: Buffer::from_host(stream, &TValue::zeros(DType::F32, size))?.value().0,
            op,
            params: default_params,
            step: 0,
            step_size: Buffer::from_host(stream, &TValue::zeros(DType::F32, 1))?.value().0,
            denom: Buffer::from_host(stream, &TValue::zeros(DType::F32, 1))?.value().0,
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

        let denom = f32::from(n_sma > params.n_sma_threshold);

        let mut blocks = Vec::new();

        let scalars = [(DValue::F32(step_size), self.step_size.clone()), (DValue::F32(denom), self.denom.clone())];

        unimplemented!();

        Ok(blocks)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        self.step = 0;
        let stream = self.momentum.creator();
        let size = self.momentum.size();
        self.momentum.copy_from_host(&stream, &TValue::zeros(DType::F32, size))?;
        self.velocity.copy_from_host(&stream, &TValue::zeros(DType::F32, size))?;
        Ok(())
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let stream = map.iter().next().unwrap().1.momentum.creator();

        let momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.momentum)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        utils::write_weights_to_file::<G>(&stream, &momentum, &format!("{path}/momentum.bin"))?;
        utils::write_weights_to_file::<G>(&stream, &velocity, &format!("{path}/velocity.bin"))?;

        let mut file = File::create(format!("{path}/step.txt")).unwrap();
        for (id, single) in map.iter() {
            writeln!(file, "{id},{}", single.step).unwrap();
        }

        Ok(())
    }

    fn load_from_checkpoint(map: &mut HashMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
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
            let stream = single.momentum.creator();
            single.momentum.copy_from_host(&stream, &TValue::F32(mom))?;
            single.velocity.copy_from_host(&stream, &TValue::F32(vel))?;
            single.step = step;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        self.params = params;

        let size = self.momentum.size();
        let device = self.momentum.creator().device();
        self.op = params.build(size).unwrap().compile(device)?;
        Ok(())
    }
}
