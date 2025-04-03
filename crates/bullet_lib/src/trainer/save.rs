use std::io::{self, Write};

use crate::nn::{Graph, Shape};

type F = fn(&Graph, &str, Vec<f32>) -> Vec<f32>;

#[derive(Clone)]
pub struct SavedFormat {
    pub(crate) id: String,
    pub(crate) quant: QuantTarget,
    pub(crate) layout: Layout,
    pub(crate) transforms: Vec<F>,
    pub(crate) round: bool,
}

impl SavedFormat {
    pub fn id(id: &str) -> Self {
        Self::new(id, QuantTarget::Float, Layout::Normal)
    }

    pub fn new(id: &str, quant: QuantTarget, layout: Layout) -> Self {
        SavedFormat { id: id.to_string(), quant, layout, transforms: Vec::new(), round: false }
    }

    pub fn round(mut self) -> Self {
        self.round = true;
        self
    }

    pub fn quantise<T: Quant>(mut self, multiplier: T::Multiplier) -> Self {
        self.quant = T::to_target(multiplier);
        self
    }

    pub fn transpose(self) -> Self {
        self.add_transform(|graph, id, weights| {
            let shape = graph.get_weights(id).shape();
            Self::transpose_impl(shape, &weights)
        })
    }

    pub fn add_transform(mut self, f: F) -> Self {
        self.transforms.push(f);
        self
    }

    pub fn write_to_byte_buffer(&self, graph: &Graph) -> io::Result<Vec<u8>> {
        let mut weights = graph.get_weights(&self.id).get_dense_vals().unwrap();

        if let Layout::Transposed(shape) = self.layout {
            assert_eq!(shape.size(), weights.len());
            weights = Self::transpose_impl(shape, &weights);
        }

        for transform in &self.transforms {
            weights = transform(graph, &self.id, weights);
        }

        self.quant.quantise(self.round, &weights)
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

    pub fn submatrix_transpose(shape: Shape, weights: &[f32]) -> Vec<f32> {
        assert_eq!(weights.len() % shape.size(), 0);

        let mut new_buf = vec![0.0; weights.len()];

        for (new, old) in new_buf.chunks_exact_mut(shape.size()).zip(weights.chunks_exact(shape.size())) {
            new.copy_from_slice(&Self::transpose_impl(shape, old));
        }

        new_buf
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Column-major
    Normal,
    /// Row-major
    Transposed(Shape),
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
    if round {
        x.round()
    } else {
        x.trunc()
    }
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
