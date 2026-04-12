pub mod builder;
pub mod rng;
pub mod save;
pub mod utils;

pub use builder::{ModelBuilder, ModelNode};

use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::{
    ir::NodeId,
    tensor::{DType, TType, TValue},
};
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    function::Function,
    runtime::{Device, Gpu, Stream},
};

pub type TensorMap<G> = BTreeMap<String, Arc<Buffer<G>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
}

impl Shape {
    pub fn new(rows: usize, cols: usize) -> Shape {
        Self { rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn size(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct Model<G: Gpu> {
    device: Arc<Device<G>>,
    weights: TensorMap<G>,
    shapes: BTreeMap<String, (Shape, Option<usize>)>,
    forward: Function<G>,
    fwd_map: BTreeMap<String, NodeId>,
    backward: Function<G>,
    bwd_map: BTreeMap<String, NodeId>,
    fwd_output_types: BTreeMap<String, TType>,
    bwd_output_types: BTreeMap<String, TType>,
}

impl<G: Gpu> Model<G> {
    pub fn device(&self) -> Arc<Device<G>> {
        self.device.clone()
    }

    pub fn weights(&self) -> &TensorMap<G> {
        &self.weights
    }

    pub fn get_weights(&self, id: impl Into<String>) -> Option<TValue> {
        self.weights.get(&id.into()).cloned().map(|tensor| tensor.clone().to_host().unwrap())
    }

    pub fn set_weights(&self, id: impl Into<String>, new_value: &TValue) -> bool {
        self.weights.get(&id.into()).cloned().map(|tensor| tensor.copy_from_host(new_value).unwrap()).is_some()
    }

    pub fn forward(
        &self,
        stream: &Arc<Stream<G>>,
        inputs: &TensorMap<G>,
        outputs: &TensorMap<G>,
    ) -> Result<SyncOnValue<G, &Function<G>>, G::Error> {
        let tensors = collect_map::<G>([("weights/", &self.weights), ("inputs/", inputs), ("", outputs)]);
        let map = self.fwd_map.iter().map(|(name, &id)| (id, tensors.get(name).unwrap().clone())).collect();
        self.forward.execute(stream.clone(), &map)
    }

    pub fn set_fwd_batch_size(&mut self, batch_size: usize) -> Result<(), G::Error> {
        self.forward.prealloc(batch_size)
    }

    pub fn set_bwd_batch_size(&mut self, batch_size: usize) -> Result<(), G::Error> {
        self.backward.prealloc(batch_size)
    }

    pub fn backward(
        &self,
        stream: &Arc<Stream<G>>,
        inputs: &TensorMap<G>,
        outputs: &TensorMap<G>,
        gradients: &TensorMap<G>,
    ) -> Result<SyncOnValue<G, &Function<G>>, G::Error> {
        let tensors = collect_map::<G>([
            ("weights/", &self.weights),
            ("inputs/", inputs),
            ("", outputs),
            ("gradients/", gradients),
        ]);
        let map = self.bwd_map.iter().map(|(name, &id)| (id, tensors.get(name).unwrap().clone())).collect();

        self.backward.execute(stream.clone(), &map)
    }

    pub fn make_gradient_tensors(&self) -> Result<TensorMap<G>, G::Error> {
        self.weights()
            .iter()
            .map(|(id, weight)| {
                Buffer::from_host(&self.device, &TValue::zeros(weight.dtype(), weight.size()))
                    .map(|buf| (id.clone(), buf))
            })
            .collect()
    }

    pub fn make_backward_output_tensors(&self) -> Result<TensorMap<G>, G::Error> {
        self.bwd_output_types
            .iter()
            .map(|(id, ty)| {
                let size = ty.size().evaluate_constant().expect("`Model` only supports constant-size outputs!");
                Buffer::from_host(&self.device, &TValue::zeros(ty.dtype(), size)).map(|buf| (id.clone(), buf))
            })
            .collect()
    }

    pub fn make_forward_output_tensors(&self, batch_size: usize) -> Result<TensorMap<G>, G::Error> {
        self.fwd_output_types
            .iter()
            .map(|(id, ty)| {
                let size = ty.size().evaluate(batch_size);
                Buffer::from_host(&self.device, &TValue::zeros(ty.dtype(), size)).map(|buf| (id.clone(), buf))
            })
            .collect()
    }

    /// Writes the weights of a graph to a file. If `gradients` is true,
    /// it will instead write the gradients of those weights.
    pub fn write_to(&self, writer: &mut impl std::io::Write) -> Result<(), G::Error> {
        let mut buf = Vec::new();

        for (id, value) in self.weights.clone() {
            if value.dtype() != DType::F32 {
                unimplemented!("Non f32 writing!");
            }

            let this_buf = value.clone().to_host()?;
            let byte_buf = utils::write_to_byte_buffer(&this_buf, &id).unwrap();
            buf.extend_from_slice(&byte_buf);
        }

        writer.write_all(&buf).unwrap();

        Ok(())
    }

    /// Loads the weights of a graph from a file. If `gradients` is true,
    /// it will instead load the gradients of those weights.
    pub fn load_from(&mut self, mut reader: impl std::io::Read) -> Result<(), G::Error> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();

        let mut offset = 0;

        while offset < buf.len() {
            let (buffer, id, bytes_read) = utils::read_from_byte_buffer(&buf[offset..]);
            let weights = self.weights.get(&id).expect("No weight with ID found!").clone();

            if weights.dtype() != DType::F32 {
                unimplemented!("Non f32 writing!");
            }

            if buffer.len() != weights.size() {
                panic!("Invalid buffer size!");
            }

            weights.copy_from_host(&TValue::F32(buffer))?;

            offset += bytes_read;
        }

        Ok(())
    }
}

fn collect_map<'a, G: Gpu + 'a>(x: impl AsRef<[(&'a str, &'a TensorMap<G>)]>) -> TensorMap<G> {
    let mut map = BTreeMap::new();

    for (pre, submap) in x.as_ref() {
        for (name, value) in submap.iter() {
            map.insert(format!("{pre}{name}"), value.clone());
        }
    }

    map
}
