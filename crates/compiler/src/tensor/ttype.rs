use std::{
    ffi::c_void,
    fmt,
    num::NonZeroUsize,
    ops::{Add, Div, Index, Mul, Sub},
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
pub struct Size(NonZeroUsize);

impl fmt::Debug for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0.get())
    }
}

impl From<usize> for Size {
    fn from(value: usize) -> Self {
        Self(value.try_into().unwrap())
    }
}

impl Size {
    pub const fn get(&self) -> usize {
        self.0.get()
    }

    pub const fn is_multiple_of(&self, size: Size) -> bool {
        self.0.get().is_multiple_of(size.0.get())
    }
}

impl Add<Size> for Size {
    type Output = Size;

    fn add(self, rhs: Size) -> Self::Output {
        Self(self.0.checked_add(rhs.0.get()).unwrap())
    }
}

impl Add<usize> for Size {
    type Output = Size;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0.checked_add(rhs).unwrap())
    }
}

impl Add<Size> for usize {
    type Output = Size;

    fn add(self, rhs: Size) -> Self::Output {
        Size(rhs.0.checked_add(self).unwrap())
    }
}

impl Sub<Size> for Size {
    type Output = Size;

    fn sub(self, rhs: Size) -> Self::Output {
        assert!(self.0 > rhs.0, "{self:?} <= {rhs:?}");
        Self((self.0.get() - rhs.0.get()).try_into().unwrap())
    }
}

impl Sub<usize> for Size {
    type Output = Size;

    fn sub(self, rhs: usize) -> Self::Output {
        assert!(self.0.get() > rhs, "{self:?} <= {rhs:?}");
        Self((self.0.get() - rhs).try_into().unwrap())
    }
}

impl Sub<Size> for usize {
    type Output = Size;

    fn sub(self, rhs: Size) -> Self::Output {
        assert!(self > rhs.0.get(), "{self:?} <= {rhs:?}");
        Size((self - rhs.0.get()).try_into().unwrap())
    }
}

impl Mul<Size> for Size {
    type Output = Size;

    fn mul(self, rhs: Size) -> Self::Output {
        Self(self.0.checked_mul(rhs.0).unwrap())
    }
}

impl Mul<usize> for Size {
    type Output = Size;

    fn mul(self, rhs: usize) -> Self::Output {
        self * Size::from(rhs)
    }
}

impl Mul<Size> for usize {
    type Output = Size;

    fn mul(self, rhs: Size) -> Self::Output {
        Size::from(self) * rhs
    }
}

impl Div<Size> for Size {
    type Output = Size;

    fn div(self, rhs: Size) -> Self::Output {
        assert!(self.is_multiple_of(rhs), "{self:?} % {rhs:?} != 0");
        Self((self.0.get() / rhs.0.get()).try_into().unwrap())
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

impl Add<Shape> for Size {
    type Output = Shape;

    fn add(self, rhs: Shape) -> Shape {
        let mut shape = Shape(vec![self]);
        shape.0.extend_from_slice(&rhs.0);
        shape
    }
}

impl Add<Shape> for usize {
    type Output = Shape;

    fn add(self, rhs: Shape) -> Shape {
        let mut shape = Shape(vec![self.into()]);
        shape.0.extend_from_slice(&rhs.0);
        shape
    }
}

impl Index<usize> for Shape {
    type Output = Size;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0[0])?;

        for x in self.0.iter().skip(1) {
            write!(f, "x{x:?}")?;
        }

        Ok(())
    }
}

impl Shape {
    pub fn size(&self) -> Size {
        let size = self.0.iter().fold(1usize, |x, &y| x * y.0.get());
        Size(NonZeroUsize::new(size).unwrap())
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

impl DType {
    pub fn bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::I32 => 4,
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

    pub fn neg_one(dtype: DType) -> Self {
        match dtype {
            DType::F32 => (-1.0).into(),
            DType::I32 => (-1).into(),
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

    pub fn ptr(&self) -> *mut c_void {
        match self {
            Self::F32(x) => (x as *const f32).cast_mut().cast(),
            Self::I32(x) => (x as *const i32).cast_mut().cast(),
        }
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
    pub fn ptr(&self) -> *const c_void {
        match self {
            Self::F32(x) => x.as_ptr().cast(),
            Self::I32(x) => x.as_ptr().cast(),
        }
    }

    pub fn mut_ptr(&mut self) -> *mut c_void {
        match self {
            Self::F32(x) => x.as_mut_ptr().cast(),
            Self::I32(x) => x.as_mut_ptr().cast(),
        }
    }

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

    pub fn f32(&self) -> &[f32] {
        if let Self::F32(x) = self { x } else { panic!("Incorrect DType!") }
    }
}
