use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

    pub fn f32(self) -> Option<f32> {
        if let Self::F32(x) = self { Some(x) } else { None }
    }

    pub fn i32(self) -> Option<i32> {
        if let Self::I32(x) = self { Some(x) } else { None }
    }
}

impl fmt::Display for DTypeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::F32(x) => write!(f, "{x}"),
            Self::I32(x) => write!(f, "{x}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DTypeTensor {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl From<DTypeValue> for DTypeTensor {
    fn from(value: DTypeValue) -> Self {
        match value {
            DTypeValue::F32(x) => DTypeTensor::F32(vec![x]),
            DTypeValue::I32(x) => DTypeTensor::I32(vec![x]),
        }
    }
}

impl DTypeTensor {
    pub fn new(dtype: DType, size: usize) -> Self {
        match dtype {
            DType::F32 => Self::F32(vec![0.0; size]),
            DType::I32 => Self::I32(vec![0; size]),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::I32(_) => DType::I32,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::F32(x) => x.len(),
            Self::I32(x) => x.len(),
        }
    }

    pub fn read(&self, idx: usize) -> DTypeValue {
        match self {
            Self::F32(x) => DTypeValue::F32(x[idx]),
            Self::I32(x) => DTypeValue::I32(x[idx]),
        }
    }

    pub fn write(&mut self, idx: usize, value: DTypeValue) {
        match (self, value) {
            (Self::F32(x), DTypeValue::F32(y)) => x[idx] = y,
            (Self::I32(x), DTypeValue::I32(y)) => x[idx] = y,
            _ => panic!(),
        }
    }
}
