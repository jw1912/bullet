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
