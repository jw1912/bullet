use std::io::{self, Write};

#[derive(Clone, Copy)]
pub enum QuantTarget {
    Float,
    I16(i16),
    I32(i32),
}

impl QuantTarget {
    pub fn quantise(self, buf: &[f32]) -> io::Result<Vec<u8>> {
        let mut quantised = Vec::<u8>::new();

        for &float in buf {
            let to_write: &[u8] = match self {
                Self::Float => &float.to_le_bytes(),
                Self::I16(q) => {
                    let x = (f32::from(q) * float) as i16;

                    if (f64::from(float) * f64::from(q)).trunc() != f64::from(x){
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Failed quantisation from f32 to i16!"));
                    }

                    &x.to_le_bytes()
                },
                Self::I32(q) => {
                    let x = (q as f32 * float) as i32;

                    if (f64::from(float) * f64::from(q)).trunc() != f64::from(x) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Failed quantisation from f32 to i32!"));
                    }

                    &x.to_le_bytes()
                },
            };

            quantised.write_all(to_write)?;
        }

        Ok(quantised)
    }
}
