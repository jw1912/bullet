use std::ops::{Add, Div, Mul, Sub};

pub trait Usable:
    Copy + Sized + Add<Self, Output = Self> + Div<Self, Output = Self> + Mul<Self, Output = Self> + Sub<Self, Output = Self>
{
}

impl Usable for f32 {}
impl Usable for i32 {}

#[derive(Clone)]
pub enum VariableExpression<T: Usable> {
    Variable(usize),
    Const(T),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
}

impl<T: Usable> VariableExpression<T> {
    pub fn evaluate(&self, variables: &[T]) -> T {
        match self {
            Self::Variable(idx) => variables[*idx],
            Self::Const(val) => *val,
            Self::Mul(x, y) => x.evaluate(variables) * y.evaluate(variables),
            Self::Div(x, y) => x.evaluate(variables) / y.evaluate(variables),
            Self::Add(x, y) => x.evaluate(variables) + y.evaluate(variables),
            Self::Sub(x, y) => x.evaluate(variables) - y.evaluate(variables),
        }
    }
}

impl<T: Usable> Mul<VariableExpression<T>> for VariableExpression<T> {
    type Output = Self;

    fn mul(self, rhs: VariableExpression<T>) -> Self::Output {
        Self::Mul(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl<T: Usable> Div<VariableExpression<T>> for VariableExpression<T> {
    type Output = Self;

    fn div(self, rhs: VariableExpression<T>) -> Self::Output {
        Self::Div(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl<T: Usable> Add<VariableExpression<T>> for VariableExpression<T> {
    type Output = Self;

    fn add(self, rhs: VariableExpression<T>) -> Self::Output {
        Self::Add(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl<T: Usable> Sub<VariableExpression<T>> for VariableExpression<T> {
    type Output = Self;

    fn sub(self, rhs: VariableExpression<T>) -> Self::Output {
        Self::Sub(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl<T: Usable> Mul<T> for VariableExpression<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Mul(Box::new(self.clone()), Box::new(VariableExpression::Const(rhs)))
    }
}

impl<T: Usable> Div<T> for VariableExpression<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::Div(Box::new(self.clone()), Box::new(VariableExpression::Const(rhs)))
    }
}

impl<T: Usable> Add<T> for VariableExpression<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self::Add(Box::new(self.clone()), Box::new(VariableExpression::Const(rhs)))
    }
}

impl<T: Usable> Sub<T> for VariableExpression<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self::Sub(Box::new(self.clone()), Box::new(VariableExpression::Const(rhs)))
    }
}
