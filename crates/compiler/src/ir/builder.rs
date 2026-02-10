use std::cell::RefCell;

use crate::ir::{
    IR, IRTrace,
    graph::{DType, DValue, GraphError, NodeId, OpType, Shape, Size, TType, TValue},
    operation::{
        BroadcastAcrossDimension, CABinary, CopyOp, NCABinary, NCABinaryOp, ReduceAcrossDimension, Reduction, Unary,
    },
};

#[derive(Default)]
pub struct IRBuilder {
    ir: RefCell<IR>,
}

impl IRBuilder {
    fn new_node<'a>(&'a self, node: NodeId) -> IRNode<'a> {
        IRNode::new(self, node)
    }

    pub fn add_op<'a>(&'a self, inputs: impl AsRef<[IRNode<'a>]>, op: impl OpType) -> Result<Vec<IRNode<'a>>, IRTrace> {
        let ids = inputs.as_ref().iter().map(IRNode::node).collect::<Vec<_>>();
        let outs = self.ir.borrow_mut().add_op(ids, Ok::<_, GraphError>(op)).unwrap();
        Ok(outs.into_iter().map(|out| self.new_node(out)).collect())
    }

    pub fn add_input<'a>(&'a self, size: impl Into<Size>, dtype: DType) -> IRNode<'a> {
        let node = self.ir.borrow_mut().add_input(TType::new(size, dtype));
        self.new_node(node)
    }

    pub fn constant<'a>(&'a self, value: TValue) -> IRNode<'a> {
        let node = self.ir.borrow_mut().add_const(value);
        self.new_node(node)
    }

    pub fn scalar<'a>(&'a self, value: impl Into<DValue>, size: impl Into<Size>) -> IRNode<'a> {
        let node = self.ir.borrow_mut().add_scalar(value, size);
        self.new_node(node)
    }

    pub fn display_ir(&self) {
        println!("{}", self.ir.borrow().graph())
    }

    pub fn build<'a>(&'a self, returns: impl AsRef<[IRNode<'a>]>) -> IR {
        let mut ir = self.ir.borrow().clone();

        for ret in returns.as_ref() {
            ir.register_output(ret.node());
        }

        ir
    }
}

#[derive(Clone, Copy)]
pub struct IRNode<'a> {
    builder: &'a IRBuilder,
    node: NodeId,
}

impl<'a> IRNode<'a> {
    pub fn new(builder: &'a IRBuilder, node: NodeId) -> Self {
        Self { builder, node }
    }

    pub fn copy(self) -> Result<Self, IRTrace> {
        self.builder.add_op([self], CopyOp(self.ty())).map(|x| x[0])
    }

    pub fn builder(self) -> &'a IRBuilder {
        self.builder
    }

    pub fn node(&self) -> NodeId {
        self.node
    }

    pub fn ty(&self) -> TType {
        self.builder.ir.borrow().get_node(self.node).unwrap().ty()
    }

    pub fn broadcast(self, shape: impl Into<Shape>, dim: usize, repeats: impl Into<Size>) -> Result<Self, IRTrace> {
        let op = BroadcastAcrossDimension::new(self.ty().dtype(), shape, dim, repeats)?;
        self.builder.add_op([self], op).map(|x| x[0])
    }

    fn reduce(self, shape: impl Into<Shape>, dim: usize, reduction: Reduction) -> Result<Self, IRTrace> {
        let op = ReduceAcrossDimension::new(self.ty().dtype(), shape, dim, reduction)?;
        self.builder.add_op([self], op).map(|x| x[0])
    }

    pub fn reduce_sum(self, shape: impl Into<Shape>, dim: usize) -> Result<Self, IRTrace> {
        self.reduce(shape, dim, Reduction::Sum)
    }

    pub fn reduce_min(self, shape: impl Into<Shape>, dim: usize) -> Result<Self, IRTrace> {
        self.reduce(shape, dim, Reduction::Min)
    }

    pub fn reduce_max(self, shape: impl Into<Shape>, dim: usize) -> Result<Self, IRTrace> {
        self.reduce(shape, dim, Reduction::Max)
    }

    pub fn pow(self, other: Self) -> Result<Self, IRTrace> {
        self.builder.add_op([self, other], NCABinaryOp::new(self.ty(), NCABinary::Pow)).map(|x| x[0])
    }

    pub fn min(self, other: Self) -> Result<Self, IRTrace> {
        self.binary(other, CABinary::Min)
    }

    pub fn max(self, other: Self) -> Result<Self, IRTrace> {
        self.binary(other, CABinary::Max)
    }

    pub fn unary(self, op: Unary) -> Result<Self, IRTrace> {
        let node = self.builder.ir.borrow_mut().add_unary(self.node, op)?;
        Ok(Self { builder: self.builder, node })
    }

    pub fn binary(self, rhs: Self, op: CABinary) -> Result<Self, IRTrace> {
        let node = self.builder.ir.borrow_mut().add_binary(self.node, rhs.node, op)?;
        Ok(Self { builder: self.builder, node })
    }
}

macro_rules! binary_impl {
    ($stdop:ident, $fnname:ident, $mapop:ident) => {
        impl<'a> std::ops::$stdop<IRNode<'a>> for IRNode<'a> {
            type Output = Result<IRNode<'a>, IRTrace>;

            fn $fnname(self, rhs: IRNode<'a>) -> Self::Output {
                self.binary(rhs, CABinary::$mapop)
            }
        }
    };
}

binary_impl!(Mul, mul, Mul);
binary_impl!(Add, add, Add);

macro_rules! unary_impl {
    ($fnname:ident, $mapop:ident) => {
        impl<'a> IRNode<'a> {
            pub fn $fnname(self) -> Result<Self, IRTrace> {
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
        impl<'a> std::ops::$stdop<$t> for IRNode<'a> {
            type Output = Result<IRNode<'a>, IRTrace>;

            fn $fnname(self, rhs: $t) -> Self::Output {
                let scalar = self.builder.scalar(rhs, self.ty().size());
                self.binary(scalar, CABinary::$mapop)
            }
        }

        impl<'a> std::ops::$stdop<IRNode<'a>> for $t {
            type Output = Result<IRNode<'a>, IRTrace>;

            fn $fnname(self, rhs: IRNode<'a>) -> Self::Output {
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

impl std::ops::Neg for IRNode<'_> {
    type Output = Result<Self, IRTrace>;

    fn neg(self) -> Self::Output {
        match self.ty().dtype() {
            DType::F32 => -1.0 * self,
            DType::I32 => -1 * self,
        }
    }
}

impl<T> std::ops::Sub<T> for IRNode<'_>
where
    Self: std::ops::Add<T, Output = Result<Self, IRTrace>>,
    T: std::ops::Neg<Output = T>,
{
    type Output = Result<Self, IRTrace>;

    fn sub(self, rhs: T) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Sub<Self> for IRNode<'_> {
    type Output = Result<Self, IRTrace>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)?
    }
}

impl<'a> std::ops::Sub<IRNode<'a>> for i32 {
    type Output = Result<IRNode<'a>, IRTrace>;

    fn sub(self, rhs: IRNode<'a>) -> Self::Output {
        self + (-rhs)?
    }
}

impl<'a> std::ops::Sub<IRNode<'a>> for f32 {
    type Output = Result<IRNode<'a>, IRTrace>;

    fn sub(self, rhs: IRNode<'a>) -> Self::Output {
        self + (-rhs)?
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl std::ops::Div<Self> for IRNode<'_> {
    type Output = Result<Self, IRTrace>;

    fn div(self, rhs: Self) -> Self::Output {
        match rhs.ty().dtype() {
            DType::F32 => self * rhs.unary(Unary::Reciprocal)?,
            DType::I32 => unimplemented!(),
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> std::ops::Div<IRNode<'a>> for f32 {
    type Output = Result<IRNode<'a>, IRTrace>;

    fn div(self, rhs: IRNode<'a>) -> Self::Output {
        self * rhs.unary(Unary::Reciprocal)?
    }
}

impl std::ops::Div<f32> for IRNode<'_> {
    type Output = Result<Self, IRTrace>;

    fn div(self, rhs: f32) -> Self::Output {
        self * (1.0 / rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_usage() -> Result<(), IRTrace> {
        let builder = IRBuilder::default();

        let x = builder.add_input(8, DType::F32);
        let a = builder.add_input(8, DType::F32);
        let b = builder.add_input(8, DType::F32);

        let y = ((a * x)? + b)?;

        let _program = builder.build([y]);

        Ok(())
    }
}
