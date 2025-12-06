use std::cell::RefCell;

use crate::{
    DType,
    elementwise::{Binary, ElementwiseDescription, ElementwiseId, Operation, Unary},
};

#[derive(Default)]
pub struct ElementwiseBuilder {
    desc: RefCell<ElementwiseDescription>,
}

impl ElementwiseBuilder {
    pub fn add_op<'a>(&'a self, op: Operation) -> Option<ElementwiseNode<'a>> {
        self.desc.borrow_mut().add_op(op).map(|node| ElementwiseNode { builder: self, node })
    }

    pub fn add_input<'a>(&'a self, dtype: DType) -> ElementwiseNode<'a> {
        ElementwiseNode { builder: self, node: self.desc.borrow_mut().add_input(dtype) }
    }

    pub fn build(self) -> ElementwiseDescription {
        self.desc.into_inner()
    }
}

#[derive(Clone, Copy)]
pub struct ElementwiseNode<'a> {
    builder: &'a ElementwiseBuilder,
    pub(crate) node: ElementwiseId,
}

impl ElementwiseNode<'_> {
    pub fn unary(self, op: Unary) -> Option<Self> {
        self.builder.add_op(Operation::Unary { input: self.node.into(), op })
    }

    pub fn binary(self, rhs: Self, op: Binary) -> Option<Self> {
        self.builder.add_op(Operation::Binary { lhs: self.node.into(), rhs: rhs.node.into(), op })
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $fn:ident) => {
        impl std::ops::$trait<Self> for ElementwiseNode<'_> {
            type Output = Self;

            fn $fn(self, rhs: Self) -> Self::Output {
                self.binary(rhs, Binary::$trait).unwrap()
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);
