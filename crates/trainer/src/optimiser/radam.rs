use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
};

use bullet_compiler::{
    ir::{
        frontend::{DType, IRBuilder, IRTrace, TValue},
        graph::DValue,
    },
    runtime::{Buffer, Device, ReadyToCompileGraph, Stream, TensorInput},
};

use super::{OptimiserState, OptimiserUpdateValue, utils};

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
    pub fn build(&self, size: usize, epsilon: f32) -> Result<ReadyToCompileGraph, IRTrace> {
        let builder = IRBuilder::default();

        // args
        let lrate = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;
        let adjus = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;
        let ssize = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;
        let denom = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;

        // inputs
        let w = builder.add_input(size, DType::F32);
        let g = builder.add_input(size, DType::F32);
        let m = builder.add_input(size, DType::F32);
        let v = builder.add_input(size, DType::F32);

        let agrd = (adjus * g)?;
        let new_m = ((self.beta1 * m)? + ((1.0 - self.beta1) * agrd)?)?;
        let new_v = ((self.beta2 * v)? + (((1.0 - self.beta2) * agrd)? * agrd)?)?;

        let val_denom = (new_m / (new_v + epsilon)?)?;
        let val = ((denom * val_denom)? + ((1.0 - denom)? * new_m)?)?;

        let mut new_w = ((w * (1.0 - (lrate * self.decay)?)?)? - ((lrate * ssize)? * val)?)?;

        if let Some((min, max)) = self.clip {
            let min = builder.scalar(DValue::F32(min), size);
            let max = builder.scalar(DValue::F32(max), size);
            new_w = new_w.min(max)?.max(min)?;
        }

        let ir = builder.build([new_w, new_m, new_v]);

        let mut tensors: HashMap<String, TensorInput> =
            [("lrate", lrate), ("adjus", adjus), ("ssize", ssize), ("denom", denom), ("g", g)]
                .iter()
                .map(|(id, node)| (id.to_string(), TensorInput::In(node.node())))
                .collect();

        tensors.insert("w".into(), TensorInput::InOut(w.node(), new_w.node()));
        tensors.insert("m".into(), TensorInput::InOut(m.node(), new_m.node()));
        tensors.insert("v".into(), TensorInput::InOut(v.node(), new_v.node()));

        ReadyToCompileGraph::new(ir, tensors)
    }
}

pub struct RAdam<D: Device> {
    momentum: Arc<D::Buffer>,
    velocity: Arc<D::Buffer>,
    op: D::CompiledGraph,
    params: RAdamParams,
    step: usize,
    step_size: Arc<D::Buffer>,
    denom: Arc<D::Buffer>,
}

impl<D: Device> OptimiserState<D> for RAdam<D> {
    type Params = RAdamParams;

    fn new(device: Arc<D>, size: usize, default_params: Self::Params) -> Result<Self, D::Error> {
        let op = device.compile(default_params.build(size, 0.00001).unwrap())?;
        let stream = device.default_stream();

        Ok(Self {
            momentum: stream.make_blocking(&TValue::zeros(DType::F32, size))?,
            velocity: stream.make_blocking(&TValue::zeros(DType::F32, size))?,
            op,
            params: default_params,
            step: 0,
            step_size: stream.make_blocking(&TValue::zeros(DType::F32, 1))?,
            denom: stream.make_blocking(&TValue::zeros(DType::F32, 1))?,
        })
    }

    fn update(
        &mut self,
        stream: &Arc<D::Stream>,
        weights: Arc<D::Buffer>,
        grads: Arc<D::Buffer>,
        gradient_factor: Arc<D::Buffer>,
        learning_rate: Arc<D::Buffer>,
    ) -> Result<OptimiserUpdateValue<D>, D::Error> {
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
        blocks.push(stream.clone().copy_scalars_nonblocking(scalars)?);

        let args = [
            ("w", weights),
            ("m", self.momentum.clone()),
            ("v", self.velocity.clone()),
            ("g", grads),
            ("adjus", gradient_factor),
            ("lrate", learning_rate),
            ("ssize", self.step_size.clone()),
            ("denom", self.denom.clone()),
        ]
        .into_iter()
        .map(|(x, y)| (x.to_string(), y))
        .collect();

        blocks.push(stream.execute_graph(&self.op, &args)?);

        Ok(blocks)
    }

    fn reset(&mut self) -> Result<(), D::Error> {
        self.step = 0;
        let stream = self.momentum.device().default_stream();
        let size = self.momentum.size();
        stream.copy_h2d_blocking(&TValue::zeros(DType::F32, size), self.momentum.clone())?;
        stream.copy_h2d_blocking(&TValue::zeros(DType::F32, size), self.velocity.clone())
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::Error> {
        let stream = map.iter().next().unwrap().1.momentum.device().default_stream();

        let momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.momentum)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        utils::write_weights_to_file::<D>(&stream, &momentum, &format!("{path}/momentum.bin"))?;
        utils::write_weights_to_file::<D>(&stream, &velocity, &format!("{path}/velocity.bin"))?;

        let mut file = File::create(format!("{path}/step.txt")).unwrap();
        for (id, single) in map.iter() {
            writeln!(file, "{id},{}", single.step).unwrap();
        }

        Ok(())
    }

    fn load_from_checkpoint(map: &mut HashMap<String, &mut Self>, path: &str) -> Result<(), D::Error> {
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
            let stream = single.momentum.device().default_stream();
            stream.copy_h2d_blocking(&TValue::F32(mom), single.momentum.clone())?;
            stream.copy_h2d_blocking(&TValue::F32(vel), single.velocity.clone())?;
            single.step = step;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), D::Error> {
        self.params = params;

        let size = self.momentum.size();
        let device = self.momentum.device();
        self.op = device.compile(params.build(size, 0.00001).unwrap())?;
        Ok(())
    }
}
