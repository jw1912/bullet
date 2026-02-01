use std::{
    fmt,
    num::NonZeroUsize,
    ops::{Add, Div, Mul, Sub},
};

/// Type of a tensor
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TType {
    size: Size,
    dtype: DType,
}

impl fmt::Debug for TType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}[{:?}]", self.dtype, self.size)
    }
}

impl TType {
    pub fn new(size: impl Into<Size>, dtype: DType) -> Self {
        Self { size: size.into(), dtype }
    }

    pub fn size(&self) -> Size {
        self.size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Size {
    var_power: u32,
    factor: NonZeroUsize,
}

impl From<usize> for Size {
    fn from(value: usize) -> Self {
        Self::constant(value)
    }
}

impl fmt::Debug for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.var_power, self.factor.get()) {
            (0, x) => write!(f, "{x}"),
            (1, 1) => write!(f, "B"),
            (1, x) => write!(f, "{x}B"),
            (x, 1) => write!(f, "B^{x}"),
            (x, y) => write!(f, "{y}B^{x}"),
        }
    }
}

impl Size {
    pub const fn variable() -> Self {
        Self { var_power: 1, factor: NonZeroUsize::new(1).unwrap() }
    }

    pub const fn constant(size: usize) -> Self {
        Self { var_power: 0, factor: NonZeroUsize::new(size).unwrap() }
    }

    pub const fn is_multiple_of(&self, size: Size) -> bool {
        self.var_power >= size.var_power && self.factor.get().is_multiple_of(size.factor.get())
    }

    pub fn is_le(&self, rhs: Self) -> bool {
        self.var_power <= rhs.var_power && self.factor <= rhs.factor
    }

    pub fn evaluate(&self, var_size: usize) -> usize {
        self.factor.get() * var_size.pow(self.var_power)
    }

    pub fn evaluate_constant(&self) -> Option<usize> {
        if self.var_power == 0 { Some(self.factor.get()) } else { None }
    }

    pub fn get_var_size(&self, size: usize) -> Option<usize> {
        if self.var_power == 0 {
            return None;
        }

        let expected = ((size / self.factor.get()) as f64).powf(1.0 / f64::from(self.var_power)) as usize;
        (self.evaluate(expected) == size).then_some(expected)
    }
}

impl Add<Size> for Size {
    type Output = Size;

    fn add(self, rhs: Size) -> Self::Output {
        assert_eq!(self.var_power, rhs.var_power, "Can only add with the same power of batch size!");
        let factor = NonZeroUsize::new(self.factor.get() + rhs.factor.get()).unwrap();
        Self { var_power: self.var_power, factor }
    }
}

impl Sub<Size> for Size {
    type Output = Size;

    fn sub(self, rhs: Size) -> Self::Output {
        assert_eq!(self.var_power, rhs.var_power, "Can only sub with the same power of batch size!");
        assert!(self.factor.get() > rhs.factor.get(), "Cannot have non-positive size!");
        let factor = NonZeroUsize::new(self.factor.get() - rhs.factor.get()).unwrap();
        Self { var_power: self.var_power, factor }
    }
}

impl<T: Into<Size>> Mul<T> for Size {
    type Output = Size;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        let factor = NonZeroUsize::new(self.factor.get() * rhs.factor.get()).unwrap();
        Self { var_power: self.var_power + rhs.var_power, factor }
    }
}

impl<T: Into<Size>> Div<T> for Size {
    type Output = Size;

    fn div(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        assert!(self.is_multiple_of(rhs), "Cannot divide by non-factor size!");
        let factor = NonZeroUsize::new(self.factor.get() / rhs.factor.get()).unwrap();
        Self { var_power: self.var_power - rhs.var_power, factor }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(Vec<Size>);

impl From<Size> for Shape {
    fn from(value: Size) -> Self {
        Self(vec![value])
    }
}

impl<T: Copy + Into<Size>> From<Vec<T>> for Shape {
    fn from(value: Vec<T>) -> Self {
        Self(value.iter().map(|&x| x.into()).collect())
    }
}

impl<T: Copy + Into<Size>, const N: usize> From<[T; N]> for Shape {
    fn from(value: [T; N]) -> Self {
        Self(value.map(|x| x.into()).into())
    }
}

impl<T: Into<Shape>> std::ops::Add<T> for Shape {
    type Output = Shape;

    fn add(mut self, rhs: T) -> Shape {
        self.0.extend_from_slice(&rhs.into().0);
        self
    }
}

impl std::ops::Add<Shape> for Size {
    type Output = Shape;

    fn add(self, rhs: Shape) -> Shape {
        let mut shape = Shape(vec![self]);
        shape.0.extend_from_slice(&rhs.0);
        shape
    }
}

impl<T: AsRef<[usize]>> std::ops::Add<T> for Size {
    type Output = Shape;

    fn add(self, rhs: T) -> Shape {
        let mut shape = Shape(vec![self]);
        let v: Vec<_> = rhs.as_ref().iter().map(|&x| x.into()).collect();
        shape.0.extend_from_slice(&v);
        shape
    }
}

impl std::ops::Add<Shape> for usize {
    type Output = Shape;

    fn add(self, rhs: Shape) -> Shape {
        let mut shape = Shape(vec![self.into()]);
        shape.0.extend_from_slice(&rhs.0);
        shape
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = Size;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0[0])?;

        for x in self.0.iter().skip(1) {
            write!(f, "x{x:?}")?;
        }

        Ok(())
    }
}

impl Shape {
    pub fn size(&self) -> Size {
        self.0.iter().fold(Size::constant(1), |x, &y| x * y)
    }

    pub fn inner(&self) -> &[Size] {
        &self.0
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }
}

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

/// Conrete value of some DType
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DValue {
    F32(f32),
    I32(i32),
}

impl From<f32> for DValue {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<i32> for DValue {
    fn from(value: i32) -> Self {
        Self::I32(value)
    }
}

impl DValue {
    pub fn zero(dtype: DType) -> Self {
        match dtype {
            DType::F32 => 0.0.into(),
            DType::I32 => 0.into(),
        }
    }

    pub fn one(dtype: DType) -> Self {
        match dtype {
            DType::F32 => 1.0.into(),
            DType::I32 => 1.into(),
        }
    }

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

impl fmt::Display for DValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::F32(x) => write!(f, "{x}"),
            Self::I32(x) => write!(f, "{x}"),
        }
    }
}

/// Concrete "Tensor" value
#[derive(Clone, Debug, PartialEq)]
pub enum TValue {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl From<DValue> for TValue {
    fn from(value: DValue) -> Self {
        match value {
            DValue::F32(x) => TValue::F32(vec![x]),
            DValue::I32(x) => TValue::I32(vec![x]),
        }
    }
}

impl TValue {
    pub fn zeros(dtype: DType, size: usize) -> Self {
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

    pub fn read(&self, idx: usize) -> DValue {
        match self {
            Self::F32(x) => DValue::F32(x[idx]),
            Self::I32(x) => DValue::I32(x[idx]),
        }
    }

    pub fn write(&mut self, idx: usize, value: DValue) {
        match (self, value) {
            (Self::F32(x), DValue::F32(y)) => x[idx] = y,
            (Self::I32(x), DValue::I32(y)) => x[idx] = y,
            _ => panic!(),
        }
    }

    pub fn scalar(&self) -> Option<DValue> {
        let val = self.read(0);

        for idx in 0..self.size() {
            if self.read(idx) != val {
                return None;
            }
        }

        Some(val)
    }
}
