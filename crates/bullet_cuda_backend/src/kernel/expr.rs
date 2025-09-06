use std::ops::{Add, Div, Mul, Sub};

pub trait Usable:
    Copy + Sized + Add<Self, Output = Self> + Div<Self, Output = Self> + Mul<Self, Output = Self> + Sub<Self, Output = Self>
{
}

impl Usable for f32 {}
impl Usable for i32 {}

#[derive(Clone, Debug)]
pub enum Expr<T: Usable> {
    Var,
    Const(T),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Min(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
}

impl Expr<i32> {
    pub fn provably_multiple_of(&self, factor: i32) -> bool {
        match self {
            Self::Const(x) => x % factor == 0,
            Self::Var => false,
            Self::Mul(x, y) => x.provably_multiple_of(factor) | y.provably_multiple_of(factor),
            Self::Div(_, _) => false,
            Self::Add(x, y) => x.provably_multiple_of(factor) & y.provably_multiple_of(factor),
            Self::Sub(x, y) => x.provably_multiple_of(factor) & y.provably_multiple_of(factor),
            Self::Min(x, y) => x.provably_multiple_of(factor) & y.provably_multiple_of(factor),
            Self::Max(x, y) => x.provably_multiple_of(factor) & y.provably_multiple_of(factor),
        }
    }
}

impl Expr<i32> {
    pub fn min(&self, other: &Self) -> Self {
        Self::Min(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn max(&self, other: &Self) -> Self {
        Self::Max(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn evaluate(&self, var: i32) -> i32 {
        match self {
            Self::Var => var,
            Self::Const(val) => *val,
            Self::Mul(x, y) => x.evaluate(var) * y.evaluate(var),
            Self::Div(x, y) => x.evaluate(var) / y.evaluate(var),
            Self::Add(x, y) => x.evaluate(var) + y.evaluate(var),
            Self::Sub(x, y) => x.evaluate(var) - y.evaluate(var),
            Self::Min(x, y) => x.evaluate(var).min(y.evaluate(var)),
            Self::Max(x, y) => x.evaluate(var).max(y.evaluate(var)),
        }
    }
}

impl Expr<f32> {
    pub fn evaluate(&self, var: f32) -> f32 {
        match self {
            Self::Var => var,
            Self::Const(val) => *val,
            Self::Mul(x, y) => x.evaluate(var) * y.evaluate(var),
            Self::Div(x, y) => x.evaluate(var) / y.evaluate(var),
            Self::Add(x, y) => x.evaluate(var) + y.evaluate(var),
            Self::Sub(x, y) => x.evaluate(var) - y.evaluate(var),
            Self::Min(x, y) => x.evaluate(var).min(y.evaluate(var)),
            Self::Max(x, y) => x.evaluate(var).max(y.evaluate(var)),
        }
    }
}

impl<T: Usable> Mul<Expr<T>> for Expr<T> {
    type Output = Self;

    fn mul(self, rhs: Expr<T>) -> Self::Output {
        Self::Mul(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl<T: Usable> Div<Expr<T>> for Expr<T> {
    type Output = Self;

    fn div(self, rhs: Expr<T>) -> Self::Output {
        Self::Div(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl<T: Usable> Add<Expr<T>> for Expr<T> {
    type Output = Self;

    fn add(self, rhs: Expr<T>) -> Self::Output {
        Self::Add(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl<T: Usable> Sub<Expr<T>> for Expr<T> {
    type Output = Self;

    fn sub(self, rhs: Expr<T>) -> Self::Output {
        Self::Sub(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl<T: Usable> Mul<T> for Expr<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Mul(Box::new(self.clone()), Box::new(Expr::Const(rhs)))
    }
}

impl<T: Usable> Div<T> for Expr<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::Div(Box::new(self.clone()), Box::new(Expr::Const(rhs)))
    }
}

impl<T: Usable> Add<T> for Expr<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self::Add(Box::new(self.clone()), Box::new(Expr::Const(rhs)))
    }
}

impl<T: Usable> Sub<T> for Expr<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self::Sub(Box::new(self.clone()), Box::new(Expr::Const(rhs)))
    }
}
