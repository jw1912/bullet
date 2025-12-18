use crate::{
    common::Shape,
    elementwise::{Binary, Unary},
    frontend::ProgramBuilder,
    ir::{
        node::IrNodeId,
        ops::{Broadcast, Reduce, ReduceOp},
    },
};

#[derive(Clone, Copy)]
pub struct ProgramNode<'a> {
    builder: &'a ProgramBuilder,
    node: IrNodeId,
}

impl<'a> ProgramNode<'a> {
    pub fn new(builder: &'a ProgramBuilder, node: IrNodeId) -> Self {
        Self { builder, node }
    }

    pub fn node(&self) -> IrNodeId {
        self.node
    }

    pub fn broadcast(self, start: impl Into<Shape>, end: impl Into<Shape>) -> Self {
        self.builder.add_op(Broadcast::new(self.node, start, end))[0]
    }

    fn reduce(self, start: impl Into<Shape>, end: impl Into<Shape>, op: ReduceOp) -> Self {
        self.builder.add_op(Reduce::new(self.node, start, end, op))[0]
    }

    pub fn reduce_sum(self, start: impl Into<Shape>, end: impl Into<Shape>) -> Self {
        self.reduce(start, end, ReduceOp::Sum)
    }

    pub fn reduce_min(self, start: impl Into<Shape>, end: impl Into<Shape>) -> Self {
        self.reduce(start, end, ReduceOp::Min)
    }

    pub fn reduce_max(self, start: impl Into<Shape>, end: impl Into<Shape>) -> Self {
        self.reduce(start, end, ReduceOp::Max)
    }

    fn unary(self, op: Unary) -> Self {
        let node = self.builder.ir.borrow_mut().modify(|inner| inner.add_unary(self.node, op)).unwrap();
        Self { builder: self.builder, node }
    }

    fn binary(self, rhs: Self, op: Binary) -> Self {
        let node = self.builder.ir.borrow_mut().modify(|inner| inner.add_binary(self.node, rhs.node, op)).unwrap();
        Self { builder: self.builder, node }
    }

    pub fn abs_pow(self, rhs: Self) -> Self {
        self.binary(rhs, Binary::AbsPow)
    }
}

macro_rules! binary_impl {
    ($stdop:ident, $fnname:ident, $mapop:ident) => {
        impl<'a> std::ops::$stdop<ProgramNode<'a>> for ProgramNode<'a> {
            type Output = ProgramNode<'a>;

            fn $fnname(self, rhs: ProgramNode<'a>) -> Self::Output {
                self.binary(rhs, Binary::$mapop)
            }
        }
    };
}

binary_impl!(Mul, mul, Mul);
binary_impl!(Add, add, Add);
binary_impl!(Sub, sub, Sub);
binary_impl!(Div, div, Div);

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
unary_impl!(log1pabs, Log1pAbs);
unary_impl!(sgn, Sgn);
unary_impl!(abs, Abs);
