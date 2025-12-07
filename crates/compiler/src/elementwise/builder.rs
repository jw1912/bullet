use std::cell::RefCell;

use crate::{
    common::{DType, DTypeValue},
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
    ($tr:ident, $fn:ident) => {
        impl std::ops::$tr<Self> for ElementwiseNode<'_> {
            type Output = Self;

            fn $fn(self, rhs: Self) -> Self::Output {
                self.binary(rhs, Binary::$tr).unwrap()
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);

macro_rules! impl_binary_const_lhs_op {
    ($tr:ident, $fn:ident) => {
        impl<'a> std::ops::$tr<ElementwiseNode<'a>> for f32 {
            type Output = ElementwiseNode<'a>;

            fn $fn(self, rhs: ElementwiseNode<'a>) -> Self::Output {
                rhs.builder
                    .add_op(Operation::Binary {
                        lhs: DTypeValue::F32(self).into(),
                        rhs: rhs.node.into(),
                        op: Binary::$tr,
                    })
                    .unwrap()
            }
        }
    };
}

impl_binary_const_lhs_op!(Add, add);
impl_binary_const_lhs_op!(Sub, sub);
impl_binary_const_lhs_op!(Mul, mul);
impl_binary_const_lhs_op!(Div, div);

macro_rules! impl_binary_const_rhs_op {
    ($tr:ident, $fn:ident) => {
        impl std::ops::$tr<f32> for ElementwiseNode<'_> {
            type Output = Self;

            fn $fn(self, rhs: f32) -> Self::Output {
                self.builder
                    .add_op(Operation::Binary {
                        lhs: self.node.into(),
                        rhs: DTypeValue::F32(rhs).into(),
                        op: Binary::$tr,
                    })
                    .unwrap()
            }
        }
    };
}

impl_binary_const_rhs_op!(Add, add);
impl_binary_const_rhs_op!(Sub, sub);
impl_binary_const_rhs_op!(Mul, mul);
impl_binary_const_rhs_op!(Div, div);

macro_rules! impl_binary_const_rhs {
    ($tr:ident, $fn:ident) => {
        impl ElementwiseNode<'_> {
            pub fn $fn(self, rhs: f32) -> Self {
                self.builder
                    .add_op(Operation::Binary {
                        lhs: self.node.into(),
                        rhs: DTypeValue::F32(rhs).into(),
                        op: Binary::$tr,
                    })
                    .unwrap()
            }
        }
    };
}

impl_binary_const_rhs!(AbsPow, abs_powf);
impl_binary_const_rhs!(Max, max);
impl_binary_const_rhs!(Min, min);
