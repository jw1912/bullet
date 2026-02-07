pub mod autograd;
pub mod builder;
pub mod utils;

pub use builder::{ModelBuilder, ModelNode};

use std::{collections::HashMap, sync::Arc};

use bullet_compiler::{
    ir::graph::{DType, TType, TValue},
    runtime::{BlockOnDrop, Buffer, Device, Stream},
};

pub type TensorMap<D> = HashMap<String, Arc<<D as Device>::Buffer>>;
type ModelAsyncReturn<D> = BlockOnDrop<<D as Device>::Stream, Vec<Arc<<D as Device>::Buffer>>>;

pub struct Model<D: Device> {
    device: Arc<D>,
    weights: TensorMap<D>,
    forward: D::CompiledGraph,
    backward: D::CompiledGraph,
    output_types: HashMap<String, TType>,
}

impl<D: Device> Model<D> {
    pub fn device(&self) -> Arc<D> {
        self.device.clone()
    }

    pub fn weights(&self) -> &TensorMap<D> {
        &self.weights
    }

    pub fn output_types(&self) -> &HashMap<String, TType> {
        &self.output_types
    }

    pub fn get_weights(&self, id: impl Into<String>) -> Option<TValue> {
        self.weights
            .get(&id.into())
            .cloned()
            .map(|tensor| self.device().default_stream().copy_d2h_blocking(tensor).unwrap())
    }

    pub fn set_weights(&self, id: impl Into<String>, new_value: &TValue) -> bool {
        self.weights
            .get(&id.into())
            .cloned()
            .map(|tensor| self.device().default_stream().copy_h2d_blocking(new_value, tensor).unwrap())
            .is_some()
    }

    pub fn forward(
        &self,
        stream: &Arc<D::Stream>,
        inputs: &TensorMap<D>,
        outputs: &TensorMap<D>,
    ) -> Result<ModelAsyncReturn<D>, D::Error> {
        let tensors = collect_map::<D>([("weights", &self.weights), ("inputs", inputs), ("outputs", outputs)]);

        stream.execute_graph(&self.forward, &tensors)
    }

    pub fn backward(
        &self,
        stream: &Arc<D::Stream>,
        inputs: &TensorMap<D>,
        outputs: &TensorMap<D>,
        gradients: &TensorMap<D>,
    ) -> Result<ModelAsyncReturn<D>, D::Error> {
        let tensors = collect_map::<D>([
            ("weights", &self.weights),
            ("inputs", inputs),
            ("outputs", outputs),
            ("gradients", gradients),
        ]);

        stream.execute_graph(&self.backward, &tensors)
    }

    pub fn make_gradient_tensors(&self, stream: &Arc<D::Stream>) -> Result<TensorMap<D>, D::Error> {
        self.weights()
            .iter()
            .map(|(id, weight)| {
                stream.clone().make_blocking(&TValue::zeros(weight.dtype(), weight.size())).map(|buf| (id.clone(), buf))
            })
            .collect()
    }

    pub fn make_output_tensors(&self, stream: &Arc<D::Stream>) -> Result<TensorMap<D>, D::Error> {
        self.output_types()
            .iter()
            .map(|(id, ty)| {
                let size = ty.size().evaluate_constant().expect("`Model` only supports constant-size outputs!");
                stream.clone().make_blocking(&TValue::zeros(ty.dtype(), size)).map(|buf| (id.clone(), buf))
            })
            .collect()
    }

    /// Writes the weights of a graph to a file. If `gradients` is true,
    /// it will instead write the gradients of those weights.
    pub fn write_to(&self, writer: &mut impl std::io::Write) -> Result<(), D::Error> {
        let mut buf = Vec::new();

        for (id, value) in self.weights.clone() {
            if value.dtype() != DType::F32 {
                unimplemented!("Non f32 writing!");
            }

            let this_buf = self.device().default_stream().copy_d2h_blocking(value.clone())?;
            let byte_buf = utils::write_to_byte_buffer(&this_buf, &id).unwrap();
            buf.extend_from_slice(&byte_buf);
        }

        writer.write_all(&buf).unwrap();

        Ok(())
    }

    /// Loads the weights of a graph from a file. If `gradients` is true,
    /// it will instead load the gradients of those weights.
    pub fn load_from(&mut self, mut reader: impl std::io::Read) -> Result<(), D::Error> {
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

            self.device().default_stream().copy_h2d_blocking(&TValue::F32(buffer), weights)?;

            offset += bytes_read;
        }

        Ok(())
    }
}

fn collect_map<'a, D: Device + 'a>(x: impl AsRef<[(&'a str, &'a TensorMap<D>)]>) -> TensorMap<D> {
    let mut map = HashMap::new();

    for (pre, submap) in x.as_ref() {
        for (name, value) in submap.iter() {
            map.insert(format!("{pre}/{name}"), value.clone());
        }
    }

    map
}
