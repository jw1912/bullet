use std::io::{self, Write};

use crate::{tensor::DenseMatrix, Shape};

#[derive(Clone)]
pub struct SavedFormat {
    pub(super) id: String,
    pub(super) quant: QuantTarget,
    pub(super) layout: Layout,
}

impl SavedFormat {
    pub fn new(id: &str, quant: QuantTarget, layout: Layout) -> Self {
        SavedFormat { id: id.to_string(), quant, layout }
    }

    pub fn write_to_byte_buffer(&self, weights: &DenseMatrix) -> io::Result<Vec<u8>> {
        let mut weight_buf = vec![0.0; weights.shape().size()];
        let written = weights.write_to_slice(&mut weight_buf);
        assert_eq!(written, weights.shape().size());

        if let Layout::Transposed = self.layout {
            weight_buf = transpose(weights.shape(), &weight_buf);
        }

        self.quant.quantise(&weight_buf)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Column-major
    Normal,
    /// Row-major
    Transposed,
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

impl QuantTarget {
    pub fn quantise(self, buf: &[f32]) -> io::Result<Vec<u8>> {
        let mut quantised = Vec::<u8>::new();

        for &float in buf {
            let to_write = match self {
                Self::Float => float.to_le_bytes().to_vec(),
                Self::I8(q) => {
                    let qf = (f64::from(q) * f64::from(float)).trunc();
                    let x = qf as i8;

                    if qf != f64::from(x) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Failed quantisation from f32 to i8!"));
                    }

                    x.to_le_bytes().to_vec()
                }
                Self::I16(q) => {
                    let qf = (f64::from(q) * f64::from(float)).trunc();
                    let x = qf as i16;

                    if qf != f64::from(x) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Failed quantisation from f32 to i16!"));
                    }

                    x.to_le_bytes().to_vec()
                }
                Self::I32(q) => {
                    let x = (q as f32 * float) as i32;

                    if (f64::from(float) * f64::from(q)).trunc() != f64::from(x) {
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

pub(super) fn transpose(shape: Shape, weights: &[f32]) -> Vec<f32> {
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
