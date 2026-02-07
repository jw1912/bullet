use std::{collections::HashMap, sync::Arc};

use bullet_compiler::{
    ir::frontend::{DType, IRBuilder, IRTrace, TValue},
    runtime::{BlockOnDrop, Buffer, Device, ReadyToCompileGraph, Stream, TensorInput},
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

pub struct AdamW<D: Device> {
    momentum: Arc<D::Buffer>,
    velocity: Arc<D::Buffer>,
    op: D::CompiledGraph,

    // params
    decay: Arc<D::Buffer>,
    beta1: Arc<D::Buffer>,
    beta2: Arc<D::Buffer>,
    min_weight: Arc<D::Buffer>,
    max_weight: Arc<D::Buffer>,
}

fn build_adam_op(size: usize, epsilon: f32) -> Result<ReadyToCompileGraph, IRTrace> {
    let builder = IRBuilder::default();

    // args
    let lrate = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;
    let adjus = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;

    // params
    let decay = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;
    let beta1 = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;
    let beta2 = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;
    let minw = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;
    let maxw = builder.add_input(1, DType::F32).broadcast([1], 0, size)?;

    // inputs
    let w = builder.add_input(size, DType::F32);
    let g = builder.add_input(size, DType::F32);
    let m = builder.add_input(size, DType::F32);
    let v = builder.add_input(size, DType::F32);

    let agrd = (adjus * g)?;
    let new_m = ((beta1 * m)? + ((1.0 - beta1)? * agrd)?)?;
    let new_v = ((beta2 * v)? + (((1.0 - beta2)? * agrd)? * agrd)?)?;

    let val = (new_m / (new_v + epsilon)?)?;

    let new_w = ((w * decay)? - (lrate * val)?)?.min(maxw)?.max(minw)?;

    let ir = builder.build([new_w, new_m, new_v]);

    let mut tensors: HashMap<String, TensorInput> = [
        ("lrate", lrate),
        ("adjus", adjus),
        ("decay", decay),
        ("beta1", beta1),
        ("beta2", beta2),
        ("minw", minw),
        ("maxw", maxw),
        ("g", g),
    ]
    .iter()
    .map(|(id, node)| (id.to_string(), TensorInput::In(node.node())))
    .collect();

    tensors.insert("w".into(), TensorInput::InOut(w.node(), new_w.node()));
    tensors.insert("m".into(), TensorInput::InOut(m.node(), new_m.node()));
    tensors.insert("v".into(), TensorInput::InOut(v.node(), new_v.node()));

    ReadyToCompileGraph::new(ir, tensors)
}

impl<D: Device> OptimiserState<D> for AdamW<D> {
    type Params = AdamWParams;

    fn new(device: Arc<D>, size: usize, default_params: Self::Params) -> Result<Self, D::Error> {
        if default_params.max_weight < default_params.min_weight {
            return Err(
                format!("Invalid clipping: {} >= {}", default_params.min_weight, default_params.max_weight).into()
            );
        }

        let op = device.compile(build_adam_op(size, 0.00001).unwrap())?;

        let stream = device.default_stream();
        let decay = stream.make_blocking(&TValue::F32(vec![default_params.decay]))?;
        let beta1 = stream.make_blocking(&TValue::F32(vec![default_params.beta1]))?;
        let beta2 = stream.make_blocking(&TValue::F32(vec![default_params.beta2]))?;
        let min_weight = stream.make_blocking(&TValue::F32(vec![default_params.min_weight]))?;
        let max_weight = stream.make_blocking(&TValue::F32(vec![default_params.max_weight]))?;

        Ok(Self {
            momentum: device.default_stream().make_blocking(&TValue::zeros(DType::F32, size))?,
            velocity: device.default_stream().make_blocking(&TValue::zeros(DType::F32, size))?,
            op,
            decay,
            beta1,
            beta2,
            min_weight,
            max_weight,
        })
    }

    fn update(
        &mut self,
        stream: &Arc<D::Stream>,
        weights: Arc<D::Buffer>,
        grads: Arc<D::Buffer>,
        gradient_factor: Arc<D::Buffer>,
        learning_rate: Arc<D::Buffer>,
    ) -> Result<BlockOnDrop<D::Stream, Vec<Arc<D::Buffer>>>, D::Error> {
        assert_eq!(weights.size(), self.momentum.size());
        assert_eq!(weights.size(), self.velocity.size());

        let args = [
            ("w", weights),
            ("m", self.momentum.clone()),
            ("v", self.velocity.clone()),
            ("g", grads),
            ("adjus", gradient_factor),
            ("lrate", learning_rate),
            ("decay", self.decay.clone()),
            ("beta1", self.beta1.clone()),
            ("beta2", self.beta2.clone()),
            ("minw", self.min_weight.clone()),
            ("maxw", self.max_weight.clone()),
        ]
        .into_iter()
        .map(|(x, y)| (x.to_string(), y))
        .collect();

        stream.execute_graph(&self.op, &args)
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

        for ((id1, mom), (id2, vel)) in momentum.into_iter().zip(velocity.into_iter()) {
            assert_eq!(id1, id2);

            let single = map.get_mut(&id1).unwrap();
            let stream = single.momentum.device().default_stream();
            stream.copy_h2d_blocking(&TValue::F32(mom), single.momentum.clone())?;
            stream.copy_h2d_blocking(&TValue::F32(vel), single.velocity.clone())?;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) {
        unimplemented!();
    }
}
