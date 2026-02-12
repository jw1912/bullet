use std::{collections::HashMap, sync::Arc};

use bullet_compiler::{
    ir::{
        frontend::{DType, IRBuilder, IRTrace, TValue},
        graph::DValue,
    },
    runtime::{Buffer, Device, ReadyToCompileGraph, Stream, TensorInput},
};

use super::{OptimiserState, OptimiserUpdateValue, utils};

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
    pub fn build(&self, size: usize) -> Result<ReadyToCompileGraph, IRTrace> {
        let builder = IRBuilder::default();

        // args
        let ilrate = builder.add_input(1, DType::F32);
        let iadjus = builder.add_input(1, DType::F32);

        let lrate = ilrate.broadcast([1], 0, size)?;
        let adjus = iadjus.broadcast([1], 0, size)?;

        // inputs
        let w = builder.add_input(size, DType::F32);
        let g = builder.add_input(size, DType::F32);
        let m = builder.add_input(size, DType::F32);
        let v = builder.add_input(size, DType::F32);

        let agrd = (adjus * g)?;
        let new_m = ((self.beta1 * m)? + ((1.0 - self.beta1) * agrd)?)?;
        let new_v = ((self.beta2 * v)? + (((1.0 - self.beta2) * agrd)? * agrd)?)?;

        let val = (new_m / (new_v.sqrt()? + 0.00000001)?)?;

        let minw = builder.scalar(DValue::F32(self.min_weight), size);
        let maxw = builder.scalar(DValue::F32(self.max_weight), size);
        let new_w = ((w * (1.0 - (lrate * self.decay)?)?)? - (lrate * val)?)?.min(maxw)?.max(minw)?;

        let ir = builder.build([new_w, new_m, new_v]);

        ReadyToCompileGraph::new(
            ir,
            [
                ("lrate".into(), TensorInput::In(ilrate.node())),
                ("adjus".into(), TensorInput::In(iadjus.node())),
                ("g".into(), TensorInput::In(g.node())),
                ("w".into(), TensorInput::InOut(w.node(), new_w.node())),
                ("m".into(), TensorInput::InOut(m.node(), new_m.node())),
                ("v".into(), TensorInput::InOut(v.node(), new_v.node())),
            ]
            .into(),
        )
    }
}

pub struct AdamW<D: Device> {
    momentum: Arc<D::Buffer>,
    velocity: Arc<D::Buffer>,
    op: D::CompiledGraph,
}

impl<D: Device> OptimiserState<D> for AdamW<D> {
    type Params = AdamWParams;

    fn new(device: Arc<D>, size: usize, default_params: Self::Params) -> Result<Self, D::Error> {
        if default_params.max_weight < default_params.min_weight {
            return Err(
                format!("Invalid clipping: {} >= {}", default_params.min_weight, default_params.max_weight).into()
            );
        }

        let op = device.compile(default_params.build(size).unwrap())?;
        let stream = device.default_stream();

        Ok(Self {
            momentum: stream.make_blocking(&TValue::zeros(DType::F32, size))?,
            velocity: stream.make_blocking(&TValue::zeros(DType::F32, size))?,
            op,
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

        let args = [
            ("w", weights),
            ("m", self.momentum.clone()),
            ("v", self.velocity.clone()),
            ("g", grads),
            ("adjus", gradient_factor),
            ("lrate", learning_rate),
        ]
        .into_iter()
        .map(|(x, y)| (x.to_string(), y))
        .collect();

        stream.execute_graph(&self.op, &args).map(|x| vec![x])
    }

    fn reset(&mut self) -> Result<(), D::Error> {
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
        utils::write_weights_to_file::<D>(&stream, &velocity, &format!("{path}/velocity.bin"))
    }

    fn load_from_checkpoint(map: &mut HashMap<String, &mut Self>, path: &str) -> Result<(), D::Error> {
        let paths = [format!("{path}/momentum.bin"), format!("{path}/velocity.bin")];
        let mut momentum = utils::load_weights_from_file(&paths[0]);
        let mut velocity = utils::load_weights_from_file(&paths[1]);

        momentum.sort_by_key(|(id, _)| id.clone());
        velocity.sort_by_key(|(id, _)| id.clone());

        for ((id1, mom), (id2, vel)) in momentum.into_iter().zip(velocity) {
            assert_eq!(id1, id2);

            let single = map.get_mut(&id1).unwrap();
            let stream = single.momentum.device().default_stream();
            stream.copy_h2d_blocking(&TValue::F32(mom), single.momentum.clone())?;
            stream.copy_h2d_blocking(&TValue::F32(vel), single.velocity.clone())?;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), D::Error> {
        let size = self.momentum.size();
        let device = self.momentum.device();
        self.op = device.compile(params.build(size).unwrap())?;
        Ok(())
    }
}
