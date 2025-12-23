use crate::{
    common::{Binary, Shape, Size, Unary},
    frontend::ProgramBuilder,
    ir::{
        node::{IrNodeId, IrType},
        operation::{BroadcastAcrossDimension, ReduceAcrossDimension, Reduction},
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

    pub fn ty(&self) -> IrType {
        self.builder.ir.borrow().get(self.node).unwrap().ty()
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
