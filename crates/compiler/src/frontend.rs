use std::cell::RefCell;

use crate::{
    core::{Binary, DType, DTypeTensor, Shape, Size, Unary},
    ir::{
        IR,
        graph::{
            IrError, IrNodeId, IrOperationType, IrType,
            operation::{BroadcastAcrossDimension, ReduceAcrossDimension, Reduction},
        },
    },
};

#[derive(Default)]
pub struct ProgramBuilder {
    ir: RefCell<IR>,
}

impl ProgramBuilder {
    fn new_node<'a>(&'a self, node: IrNodeId) -> ProgramNode<'a> {
        ProgramNode::new(self, node)
    }

    pub fn add_op<'a>(
        &'a self,
        inputs: impl AsRef<[ProgramNode<'a>]>,
        op: impl IrOperationType,
    ) -> Vec<ProgramNode<'a>> {
        let ids = inputs.as_ref().iter().map(ProgramNode::node).collect::<Vec<_>>();
        let outs = self.ir.borrow_mut().add_op(ids, Ok::<_, IrError>(op)).unwrap();
        outs.into_iter().map(|out| self.new_node(out)).collect()
    }

    pub fn add_leaf<'a>(&'a self, size: impl Into<Size>, dtype: DType) -> ProgramNode<'a> {
        let node = self.ir.borrow_mut().add_leaf(IrType::new(size, dtype));
        self.new_node(node)
    }

    pub fn constant<'a>(&'a self, value: DTypeTensor) -> ProgramNode<'a> {
        let node = self.ir.borrow_mut().add_const(value);
        self.new_node(node)
    }

    pub fn display_ir(&self) {
        println!("{}", self.ir.borrow().graph().as_highlighted())
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

    fn unary(self, op: Unary) -> Self {
        let node = self.builder.ir.borrow_mut().add_unary(self.node, op).unwrap();
        Self { builder: self.builder, node }
    }

    fn binary(self, rhs: Self, op: Binary) -> Self {
        let node = self.builder.ir.borrow_mut().add_binary(self.node, rhs.node, op).unwrap();
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

macro_rules! binary_const_impl {
    ($stdop:ident, $fnname:ident, $mapop:ident, $t:ty) => {
        impl<'a> std::ops::$stdop<$t> for ProgramNode<'a> {
            type Output = ProgramNode<'a>;

            fn $fnname(self, rhs: $t) -> Self::Output {
                self.unary(Unary::BinaryWithConst { op: Binary::$mapop, val: rhs.into(), lhs: true })
            }
        }

        impl<'a> std::ops::$stdop<ProgramNode<'a>> for $t {
            type Output = ProgramNode<'a>;

            fn $fnname(self, rhs: ProgramNode<'a>) -> Self::Output {
                rhs.unary(Unary::BinaryWithConst { op: Binary::$mapop, val: self.into(), lhs: false })
            }
        }
    };
}

binary_const_impl!(Mul, mul, Mul, f32);
binary_const_impl!(Add, add, Add, f32);
binary_const_impl!(Sub, sub, Sub, f32);
binary_const_impl!(Div, div, Div, f32);

binary_const_impl!(Mul, mul, Mul, i32);
binary_const_impl!(Add, add, Add, i32);
binary_const_impl!(Sub, sub, Sub, i32);
binary_const_impl!(Div, div, Div, i32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_usage() {
        let builder = ProgramBuilder::default();

        let x = builder.add_leaf(8, DType::F32);
        let a = builder.add_leaf(8, DType::F32);
        let b = builder.add_leaf(8, DType::F32);

        let y = a * x + b;

        let _program = builder.build([y]);
    }
}
