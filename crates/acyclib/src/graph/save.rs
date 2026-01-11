use std::{
    collections::HashMap,
    io::{self, Write},
    rc::Rc,
};

use crate::{
    device::Device,
    graph::{Graph, Shape},
};

#[derive(Clone)]
pub struct TensorStore {
    pub values: Vec<f32>,
    pub shape: Shape,
}

impl TensorStore {
    #[deprecated(note = "You can access `.values` directly!")]
    pub fn get_dense_vals(&self) -> Option<Vec<f32>> {
        Some(self.values.clone())
    }
}

pub struct GraphWeights {
    stores: HashMap<String, TensorStore>,
}

impl<D: Device> From<&Graph<D>> for GraphWeights {
    fn from(graph: &Graph<D>) -> Self {
        let ids = graph.weight_ids();

        let mut stores = HashMap::new();

        for id in ids {
            let weight = graph.get_weights(&id);
            let values = weight.get_dense_vals().unwrap();
            let shape = weight.shape();
            let existing = stores.insert(id, TensorStore { values, shape });
            assert!(existing.is_none(), "Duplicate weight IDs in graph?!?");
        }

        Self { stores }
    }
}

impl GraphWeights {
    pub fn get(&self, id: &str) -> TensorStore {
        self.stores.get(id).cloned().unwrap()
    }

    #[deprecated(note = "Use `.get` instead!")]
    pub fn get_weights(&self, id: &str) -> TensorStore {
        self.get(id)
    }
}

type Transform = Rc<dyn Fn(&GraphWeights, Vec<f32>) -> Vec<f32>>;

#[derive(Clone)]
pub struct SavedFormat {
    custom: Option<Vec<u8>>,
    quant: QuantTarget,
    transforms: Vec<Transform>,
    round: bool,
    id: Option<String>,
}

impl SavedFormat {
    /// Save a custom set of bytes.
    /// This should be used to add a network header, padding, etc.
    pub fn custom(bytes: impl Into<Vec<u8>>) -> Self {
        Self { custom: Some(bytes.into()), ..Self::empty() }
    }

    pub fn get_id(&self) -> Option<String> {
        self.id.clone()
    }

    /// Create a `SavedFormat` that is initialised with the weights from `id`.
    pub fn id(id: &str) -> Self {
        let id = id.to_string();
        Self { id: Some(id.clone()), ..Self::empty() }.transform(move |store, _| store.get(&id).values)
    }

    /// Create an empty `SavedFormat`, where the initial values are empty.
    /// Appropriate for constructing save formats where multiple weights are interleaved.
    pub fn empty() -> Self {
        SavedFormat { custom: None, id: None, quant: QuantTarget::Float, transforms: Vec::new(), round: false }
    }

    /// If quantising, round rather than truncate.
    pub fn round(mut self) -> Self {
        assert!(self.custom.is_none());
        self.round = true;
        self
    }

    /// Write weights quantised by factor `multiplier` as type `T`.
    pub fn quantise<T: Quant>(mut self, multiplier: T::Multiplier) -> Self {
        assert!(self.custom.is_none());
        self.quant = T::to_target(multiplier);
        self
    }

    /// Transpose current values using the shape of the weights from weight `id`.
    /// Panics if this `SavedFormat` was constructed without an associated weight `id`.
    pub fn transpose(self) -> Self {
        let id = self.id.clone().unwrap();
        self.transform(move |graph, weights| Self::transpose_impl(graph.get(&id).shape, &weights))
    }

    /// Transform current values using the provided closure.
    pub fn transform(mut self, f: impl Fn(&GraphWeights, Vec<f32>) -> Vec<f32> + 'static) -> Self {
        assert!(self.custom.is_none());
        self.transforms.push(Rc::new(f));
        self
    }

    #[deprecated(note = "Use `.transform(|store, mut values| { ... })` instead!")]
    pub fn add_transform(mut self, f: impl Fn(&GraphWeights, &str, Vec<f32>) -> Vec<f32> + 'static) -> Self {
        assert!(self.custom.is_none());
        let id = self.get_id().unwrap();
        self.transforms.push(Rc::new(move |store, vals| f(store, &id, vals)));
        self
    }

    pub fn write_to_byte_buffer(&self, graph: &GraphWeights) -> io::Result<Vec<u8>> {
        match &self.custom {
            Some(bytes) => Ok(bytes.clone()),
            None => {
                let mut weights = Vec::new();

                for transform in &self.transforms {
                    weights = transform(graph, weights);
                }

                self.quant.quantise(self.round, &weights)
            }
        }
    }

    /// Read quantized bytes back to f32 values.
    ///
    /// This is the inverse of `write_to_byte_buffer` for the quantization step only.
    /// Note that transforms (e.g., transpose) are NOT reversed; if you need to undo
    /// transformations, you must apply them manually after reading.
    ///
    /// For custom byte buffers, this returns an error since there is no defined
    /// dequantization for arbitrary custom data.
    pub fn read_from_byte_buffer(&self, bytes: &[u8]) -> io::Result<Vec<f32>> {
        if self.custom.is_some() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Cannot dequantize custom byte buffers"));
        }

        self.quant.dequantise(bytes)
    }

    pub(crate) fn transpose_impl(shape: Shape, weights: &[f32]) -> Vec<f32> {
        assert_eq!(shape.size(), weights.len());

        let rows = shape.rows();
        let cols = shape.cols();
        let mut new_buf = vec![0.0; shape.size()];

        for i in 0..rows {
            for j in 0..cols {
                new_buf[cols * i + j] = weights[rows * j + i];
            }
        }

        new_buf
    }
}

#[derive(Clone, Copy)]
pub enum QuantTarget {
    Float,
    /// This takes an `i16` because it is common to want to use a quantisation
    /// value of, say, 128 with weights clipped to [-0.99, 0.99]
    I8(i16),
    I16(i16),
    I32(i32),
}

fn round_or_trunc(x: f64, round: bool) -> f64 {
    if round { x.round() } else { x.trunc() }
}

impl QuantTarget {
    /// Returns the size in bytes of a single element for this quantization target.
    pub fn element_size(self) -> usize {
        match self {
            Self::Float => 4,
            Self::I8(_) => 1,
            Self::I16(_) => 2,
            Self::I32(_) => 4,
        }
    }

    /// Dequantize bytes back to f32 values.
    ///
    /// This is the inverse of `quantise`: it reads the quantized byte buffer
    /// and converts it back to f32 values by dividing by the quantization factor.
    pub fn dequantise(self, bytes: &[u8]) -> io::Result<Vec<f32>> {
        let elem_size = self.element_size();

        if bytes.len() % elem_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Byte buffer length {} is not a multiple of element size {}", bytes.len(), elem_size),
            ));
        }

        let num_elements = bytes.len() / elem_size;
        let mut result = Vec::with_capacity(num_elements);

        match self {
            Self::Float => {
                for chunk in bytes.chunks_exact(4) {
                    let arr: [u8; 4] = chunk.try_into().unwrap();
                    result.push(f32::from_le_bytes(arr));
                }
            }
            Self::I8(q) => {
                let q_f32 = f32::from(q);
                for &byte in bytes {
                    let val = byte as i8;
                    result.push(f32::from(val) / q_f32);
                }
            }
            Self::I16(q) => {
                let q_f32 = f32::from(q);
                for chunk in bytes.chunks_exact(2) {
                    let arr: [u8; 2] = chunk.try_into().unwrap();
                    let val = i16::from_le_bytes(arr);
                    result.push(f32::from(val) / q_f32);
                }
            }
            Self::I32(q) => {
                let q_f32 = q as f32;
                for chunk in bytes.chunks_exact(4) {
                    let arr: [u8; 4] = chunk.try_into().unwrap();
                    let val = i32::from_le_bytes(arr);
                    result.push(val as f32 / q_f32);
                }
            }
        }

        Ok(result)
    }

    pub fn quantise(self, round: bool, buf: &[f32]) -> io::Result<Vec<u8>> {
        let mut quantised = Vec::<u8>::new();

        for &float in buf {
            let to_write = match self {
                Self::Float => float.to_le_bytes().to_vec(),
                Self::I8(q) => {
                    let qf = round_or_trunc(f64::from(q) * f64::from(float), round);
                    let x = qf as i8;

                    if qf != f64::from(x) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Failed quantisation from f32 to i8!"));
                    }

                    x.to_le_bytes().to_vec()
                }
                Self::I16(q) => {
                    let qf = round_or_trunc(f64::from(q) * f64::from(float), round);
                    let x = qf as i16;

                    if qf != f64::from(x) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Failed quantisation from f32 to i16!"));
                    }

                    x.to_le_bytes().to_vec()
                }
                Self::I32(q) => {
                    let qf = round_or_trunc(f64::from(q) * f64::from(float), round);
                    let x = qf as i32;

                    if qf != f64::from(x) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Failed quantisation from f32 to i32!"));
                    }

                    x.to_le_bytes().to_vec()
                }
            };

            quantised.write_all(&to_write)?;
        }

        Ok(quantised)
    }
}

pub trait Quant {
    type Multiplier;

    fn to_target(q: Self::Multiplier) -> QuantTarget;
}

impl Quant for i8 {
    type Multiplier = i16;

    fn to_target(q: Self::Multiplier) -> QuantTarget {
        QuantTarget::I8(q)
    }
}

impl Quant for i16 {
    type Multiplier = i16;

    fn to_target(q: Self::Multiplier) -> QuantTarget {
        QuantTarget::I16(q)
    }
}

impl Quant for i32 {
    type Multiplier = i32;

    fn to_target(q: Self::Multiplier) -> QuantTarget {
        QuantTarget::I32(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn test_dequantise_float() {
        let values = vec![1.5f32, -2.25, 0.0, 3.125];
        let quant = QuantTarget::Float;

        let bytes = quant.quantise(false, &values).unwrap();
        let recovered = quant.dequantise(&bytes).unwrap();

        assert_eq!(values.len(), recovered.len());
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert_eq!(*orig, *rec);
        }
    }

    #[test]
    fn test_dequantise_i8() {
        let values = vec![0.5f32, -0.25, 0.0, 0.75];
        let quant = QuantTarget::I8(128);

        let bytes = quant.quantise(true, &values).unwrap();
        let recovered = quant.dequantise(&bytes).unwrap();

        assert_eq!(values.len(), recovered.len());
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            // i8 quantization has limited precision
            assert!(approx_eq(*orig, *rec, 1.0 / 128.0));
        }
    }

    #[test]
    fn test_dequantise_i16() {
        let values = vec![0.5f32, -0.25, 0.0, 0.999];
        let quant = QuantTarget::I16(1024);

        let bytes = quant.quantise(true, &values).unwrap();
        let recovered = quant.dequantise(&bytes).unwrap();

        assert_eq!(values.len(), recovered.len());
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            // i16 quantization has better precision
            assert!(approx_eq(*orig, *rec, 1.0 / 1024.0));
        }
    }

    #[test]
    fn test_dequantise_i32() {
        let values = vec![100.5f32, -50.25, 0.0, 1000.125];
        let quant = QuantTarget::I32(256);

        let bytes = quant.quantise(true, &values).unwrap();
        let recovered = quant.dequantise(&bytes).unwrap();

        assert_eq!(values.len(), recovered.len());
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!(approx_eq(*orig, *rec, 1.0 / 256.0));
        }
    }

    #[test]
    fn test_dequantise_invalid_length() {
        let quant = QuantTarget::I16(128);
        // 3 bytes is not a multiple of 2 (i16 size)
        let invalid_bytes = vec![0u8, 1, 2];

        let result = quant.dequantise(&invalid_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_element_size() {
        assert_eq!(QuantTarget::Float.element_size(), 4);
        assert_eq!(QuantTarget::I8(128).element_size(), 1);
        assert_eq!(QuantTarget::I16(256).element_size(), 2);
        assert_eq!(QuantTarget::I32(512).element_size(), 4);
    }

    #[test]
    fn test_saved_format_read_from_byte_buffer() {
        let format = SavedFormat::empty().quantise::<i16>(512);
        let values = vec![0.5f32, -0.25, 0.125];

        // Manually quantize to simulate reading from a file
        let bytes = QuantTarget::I16(512).quantise(false, &values).unwrap();
        let recovered = format.read_from_byte_buffer(&bytes).unwrap();

        assert_eq!(values.len(), recovered.len());
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!(approx_eq(*orig, *rec, 1.0 / 512.0));
        }
    }

    #[test]
    fn test_saved_format_read_custom_error() {
        let format = SavedFormat::custom(vec![1, 2, 3]);

        let result = format.read_from_byte_buffer(&[1, 2, 3]);
        assert!(result.is_err());
    }
}
