use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    I32,
}

impl fmt::Debug for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::F32 => write!(f, "f32"),
            Self::I32 => write!(f, "i32"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DTypeValue {
    F32(f32),
    I32(i32),
}

impl DTypeValue {
    pub fn dtype(&self) -> DType {
        match *self {
            Self::F32(_) => DType::F32,
            Self::I32(_) => DType::I32,
        }
    }
}
