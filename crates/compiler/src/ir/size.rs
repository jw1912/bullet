use std::{
    fmt,
    num::NonZeroUsize,
    ops::{Add, Div, Mul, Sub},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Size {
    batch_power: usize,
    factor: NonZeroUsize,
}

impl From<usize> for Size {
    fn from(value: usize) -> Self {
        Self::constant(value)
    }
}

impl fmt::Debug for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.batch_power, self.factor.get()) {
            (0, x) => write!(f, "{x}"),
            (1, 1) => write!(f, "B"),
            (1, x) => write!(f, "{x}B"),
            (x, 1) => write!(f, "B^{x}"),
            (x, y) => write!(f, "{y}B^{x}"),
        }
    }
}

impl Size {
    pub const fn batched() -> Self {
        Self { batch_power: 1, factor: NonZeroUsize::new(1).unwrap() }
    }

    pub const fn constant(size: usize) -> Self {
        Self { batch_power: 0, factor: NonZeroUsize::new(size).unwrap() }
    }

    pub const fn is_multiple_of(&self, size: Size) -> bool {
        self.batch_power >= size.batch_power && self.factor.get().is_multiple_of(size.factor.get())
    }
}

impl Add<Size> for Size {
    type Output = Size;

    fn add(self, rhs: Size) -> Self::Output {
        assert_eq!(self.batch_power, rhs.batch_power, "Can only add with the same power of batch size!");
        let factor = NonZeroUsize::new(self.factor.get() + rhs.factor.get()).unwrap();
        Self { batch_power: self.batch_power, factor }
    }
}

impl Sub<Size> for Size {
    type Output = Size;

    fn sub(self, rhs: Size) -> Self::Output {
        assert_eq!(self.batch_power, rhs.batch_power, "Can only sub with the same power of batch size!");
        assert!(self.factor.get() > rhs.factor.get(), "Cannot have non-positive size!");
        let factor = NonZeroUsize::new(self.factor.get() - rhs.factor.get()).unwrap();
        Self { batch_power: self.batch_power, factor }
    }
}

impl<T: Into<Size>> Mul<T> for Size {
    type Output = Size;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        let factor = NonZeroUsize::new(self.factor.get() * rhs.factor.get()).unwrap();
        Self { batch_power: self.batch_power + rhs.batch_power, factor }
    }
}

impl<T: Into<Size>> Div<T> for Size {
    type Output = Size;

    fn div(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        assert!(self.is_multiple_of(rhs), "Cannot divide by non-factor size!");
        let factor = NonZeroUsize::new(self.factor.get() / rhs.factor.get()).unwrap();
        Self { batch_power: self.batch_power - rhs.batch_power, factor }
    }
}
