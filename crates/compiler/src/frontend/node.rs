use crate::{
    frontend::ProgramBuilder,
    ir::{
        node::IrNodeId,
        ops::{Broadcast, Reduce, ReduceOp, Shape},
    },
    map::{BinaryOp, MapOp, UnaryOp},
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

    pub fn reduce(self, start: impl Into<Shape>, end: impl Into<Shape>, op: ReduceOp) -> Self {
        self.builder.add_op(Reduce::new(self.node, start, end, op))[0]
    }

    fn unary(self, op: UnaryOp) -> Self {
        self.builder.add_op(MapOp::Unary { inp: self.node, op })[0]
    }

    fn binary(self, rhs: Self, op: BinaryOp) -> Self {
        self.builder.add_op(MapOp::Binary { lhs: self.node, rhs: rhs.node, op })[0]
    }

    pub fn abs_pow(self, rhs: Self) -> Self {
        self.binary(rhs, BinaryOp::AbsPow)
    }
}

macro_rules! binary_impl {
    ($stdop:ident, $fnname:ident, $mapop:ident) => {
        impl<'a> std::ops::$stdop<ProgramNode<'a>> for ProgramNode<'a> {
            type Output = ProgramNode<'a>;

            fn $fnname(self, rhs: ProgramNode<'a>) -> Self::Output {
                self.binary(rhs, BinaryOp::$mapop)
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
                self.unary(UnaryOp::$mapop)
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
