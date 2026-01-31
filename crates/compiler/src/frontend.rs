use std::cell::RefCell;

use crate::{
    IR,
    graph::{DType, DValue, GraphError, NodeId, OpType, Shape, Size, TType, TValue},
    operation::{BroadcastAcrossDimension, CABinary, ReduceAcrossDimension, Reduction, Unary},
};

#[derive(Default)]
pub struct ProgramBuilder {
    ir: RefCell<IR>,
}

impl ProgramBuilder {
    fn new_node<'a>(&'a self, node: NodeId) -> ProgramNode<'a> {
        ProgramNode::new(self, node)
    }

    pub fn add_op<'a>(&'a self, inputs: impl AsRef<[ProgramNode<'a>]>, op: impl OpType) -> Vec<ProgramNode<'a>> {
        let ids = inputs.as_ref().iter().map(ProgramNode::node).collect::<Vec<_>>();
        let outs = self.ir.borrow_mut().add_op(ids, Ok::<_, GraphError>(op)).unwrap();
        outs.into_iter().map(|out| self.new_node(out)).collect()
    }

    pub fn add_input<'a>(&'a self, size: impl Into<Size>, dtype: DType) -> ProgramNode<'a> {
        let node = self.ir.borrow_mut().add_input(TType::new(size, dtype));
        self.new_node(node)
    }

    pub fn constant<'a>(&'a self, value: TValue) -> ProgramNode<'a> {
        let node = self.ir.borrow_mut().add_const(value);
        self.new_node(node)
    }

    pub fn scalar<'a>(&'a self, value: impl Into<DValue>, size: impl Into<Size>) -> ProgramNode<'a> {
        let node = self.ir.borrow_mut().add_scalar(value, size);
        self.new_node(node)
    }

    pub fn display_ir(&self) {
        println!("{}", self.ir.borrow().graph())
    }

    pub fn build<'a>(&'a self, returns: impl AsRef<[ProgramNode<'a>]>) -> IR {
        let mut ir = self.ir.borrow().clone();

        for ret in returns.as_ref() {
            ir.register_output(ret.node());
        }

        ir
    }
}

#[derive(Clone, Copy)]
pub struct ProgramNode<'a> {
    builder: &'a ProgramBuilder,
    node: NodeId,
}

impl<'a> ProgramNode<'a> {
    pub fn new(builder: &'a ProgramBuilder, node: NodeId) -> Self {
        Self { builder, node }
    }

    pub fn make_scalar(&self, value: impl Into<DValue>, size: impl Into<Size>) -> Self {
        self.builder.scalar(value, size)
    }

    pub fn node(&self) -> NodeId {
        self.node
    }

    pub fn ty(&self) -> TType {
        self.builder.ir.borrow().get_node(self.node).unwrap().ty()
    }

    pub fn broadcast(self, shape: impl Into<Shape>, dim: usize, repeats: impl Into<Size>) -> Self {
        let op = BroadcastAcrossDimension::new(self.ty().dtype(), shape, dim, repeats).unwrap();
        self.builder.add_op([self], op)[0]
    }

    fn reduce(self, shape: impl Into<Shape>, dim: usize, reduction: Reduction) -> Self {
        let op = ReduceAcrossDimension::new(self.ty().dtype(), shape, dim, reduction).unwrap();
        self.builder.add_op([self], op)[0]
    }

    pub fn reduce_sum(self, shape: impl Into<Shape>, dim: usize) -> Self {
        self.reduce(shape, dim, Reduction::Sum)
    }

    pub fn reduce_min(self, shape: impl Into<Shape>, dim: usize) -> Self {
        self.reduce(shape, dim, Reduction::Min)
    }

    pub fn reduce_max(self, shape: impl Into<Shape>, dim: usize) -> Self {
        self.reduce(shape, dim, Reduction::Max)
    }

    pub fn unary(self, op: Unary) -> Self {
        let node = self.builder.ir.borrow_mut().add_unary(self.node, op).unwrap();
        Self { builder: self.builder, node }
    }

    pub fn binary(self, rhs: Self, op: CABinary) -> Self {
        let node = self.builder.ir.borrow_mut().add_binary(self.node, rhs.node, op).unwrap();
        Self { builder: self.builder, node }
    }
}

macro_rules! binary_impl {
    ($stdop:ident, $fnname:ident, $mapop:ident) => {
        impl<'a> std::ops::$stdop<ProgramNode<'a>> for ProgramNode<'a> {
            type Output = ProgramNode<'a>;

            fn $fnname(self, rhs: ProgramNode<'a>) -> Self::Output {
                self.binary(rhs, CABinary::$mapop)
            }
        }
    };
}

binary_impl!(Mul, mul, Mul);
binary_impl!(Add, add, Add);

macro_rules! unary_impl {
    ($fnname:ident, $mapop:ident) => {
        impl<'a> ProgramNode<'a> {
            pub fn $fnname(self) -> Self {
                self.unary(Unary::$mapop)
            }
        }
    };
}

unary_impl!(sin, Sin);
unary_impl!(cos, Cos);
unary_impl!(tan, Tan);
unary_impl!(sinh, Sinh);
unary_impl!(cosh, Cosh);
unary_impl!(tanh, Tanh);
unary_impl!(exp, Exp);
unary_impl!(log, Log);
unary_impl!(sgn, Sgn);
unary_impl!(abs, Abs);

macro_rules! binary_const_impl {
    ($stdop:ident, $fnname:ident, $mapop:ident, $t:ty) => {
        impl<'a> std::ops::$stdop<$t> for ProgramNode<'a> {
            type Output = ProgramNode<'a>;

            fn $fnname(self, rhs: $t) -> Self::Output {
                let scalar = self.builder.scalar(rhs, self.ty().size());
                self.binary(scalar, CABinary::$mapop)
            }
        }

        impl<'a> std::ops::$stdop<ProgramNode<'a>> for $t {
            type Output = ProgramNode<'a>;

            fn $fnname(self, rhs: ProgramNode<'a>) -> Self::Output {
                let lhs = rhs.builder.scalar(self, rhs.ty().size());
                lhs.binary(rhs, CABinary::$mapop)
            }
        }
    };
}

binary_const_impl!(Mul, mul, Mul, f32);
binary_const_impl!(Add, add, Add, f32);
binary_const_impl!(Mul, mul, Mul, i32);
binary_const_impl!(Add, add, Add, i32);

impl std::ops::Neg for ProgramNode<'_> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self.ty().dtype() {
            DType::F32 => -1.0 * self,
            DType::I32 => -1 * self,
        }
    }
}

impl<T> std::ops::Sub<T> for ProgramNode<'_>
where
    Self: std::ops::Add<T, Output = Self>,
    T: std::ops::Neg<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self {
        self + (-rhs)
    }
}

impl<'a> std::ops::Sub<ProgramNode<'a>> for i32 {
    type Output = ProgramNode<'a>;

    fn sub(self, rhs: ProgramNode<'a>) -> Self::Output {
        self + (-rhs)
    }
}

impl<'a> std::ops::Sub<ProgramNode<'a>> for f32 {
    type Output = ProgramNode<'a>;

    fn sub(self, rhs: ProgramNode<'a>) -> Self::Output {
        self + (-rhs)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl std::ops::Div<Self> for ProgramNode<'_> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        match rhs.ty().dtype() {
            DType::F32 => self * rhs.unary(Unary::Reciprocal),
            DType::I32 => unimplemented!(),
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> std::ops::Div<ProgramNode<'a>> for f32 {
    type Output = ProgramNode<'a>;

    fn div(self, rhs: ProgramNode<'a>) -> Self::Output {
        self * rhs.unary(Unary::Reciprocal)
    }
}

impl std::ops::Div<f32> for ProgramNode<'_> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        self * (1.0 / rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_usage() {
        let builder = ProgramBuilder::default();

        let x = builder.add_input(8, DType::F32);
        let a = builder.add_input(8, DType::F32);
        let b = builder.add_input(8, DType::F32);

        let y = a * x + b;

        let _program = builder.build([y]);
    }
}
