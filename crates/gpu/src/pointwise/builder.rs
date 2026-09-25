use std::{
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{Mutex, MutexGuard},
};

use bullet_compiler::{
    ir::{IRError, NodeId},
    tensor::{
        DType, DValue, Size, TType,
        operation::{CABinary, SparseMatmul, Unary},
    },
};

use crate::pointwise::{PointwiseIR, operations::PType};

pub trait CoercesToPointwiseNode<'a> {
    /// Coerce into a node that is valid as an operand alongside a node of type `ty`,
    /// i.e. scalar literals become constants of matching dtype and vector width.
    fn coerce(&self, builder: &'a PointwiseBuilder, ty: PType) -> PointwiseNode<'a>;
}

impl<'a> CoercesToPointwiseNode<'a> for PointwiseNode<'a> {
    fn coerce(&self, _: &'a PointwiseBuilder, ty: PType) -> PointwiseNode<'a> {
        match (self.p2size(), p2size_of(ty)) {
            (0, p2size) if p2size > 0 => self.broadcast(p2size),
            _ => *self,
        }
    }
}

fn p2size_of(ty: PType) -> u8 {
    match ty {
        PType::Variable { p2size, .. } => p2size,
        PType::Pointer(_) => 0,
    }
}

/// Builder for [`PointwiseIR`], the IR describing the body of a single kernel.
///
/// Unlike the model level IR, every node here is a value local to one thread, so
/// the "shape" of a node is just its dtype plus the number of elements each thread
/// handles at once - `2^p2size` of them. All ops on a kernel operate at the same
/// vector width, so the builder carries it and applies it by default, only leaving
/// it explicit where index arithmetic forces scalar operands.
pub struct PointwiseBuilder {
    ir: Mutex<PointwiseIR>,
    //p2size: u8,
}

impl PointwiseBuilder {
    /// `size` is the number of threads the kernel is launched with, and each of them
    /// processes `2^p2size` elements at a time.
    pub fn new(size: impl Into<Size>) -> Self {
        Self { ir: Mutex::new(PointwiseIR::new(size.into()).unwrap()) }
    }

    pub fn ir(&'_ self) -> MutexGuard<'_, PointwiseIR> {
        self.ir.try_lock().unwrap()
    }

    pub fn inner(&self) -> PointwiseIR {
        self.ir().clone()
    }

    /// Number of threads the kernel is launched with
    pub fn size(&self) -> Size {
        self.ir().size()
    }

    fn add(&self, f: impl FnOnce(&mut PointwiseIR) -> Result<NodeId, IRError>) -> NodeId {
        f(&mut self.ir()).unwrap()
    }

    /// The index of the current thread, as a scalar integer
    pub fn tid(&'_ self) -> PointwiseNode<'_> {
        PointwiseNode { builder: self, node: self.ir().tid() }
    }

    /// New kernel argument, which becomes an input or an output of the kernel
    /// depending on whether it is read from or written to
    pub fn new_buffer(&'_ self, ty: TType) -> PointwiseBuf<'_> {
        PointwiseBuf { builder: self, node: self.ir().add_buf(ty) }
    }

    pub fn new_constant(&'_ self, value: impl Into<DValue>, p2size: u8) -> PointwiseNode<'_> {
        let value = value.into();
        PointwiseNode { builder: self, node: self.ir().add_const(value, p2size) }
    }
}

/// Kernel argument. Reading from it makes it an input of the kernel and writing
/// to it makes it an output, doing neither is an error when lowering.
#[derive(Clone, Copy)]
pub struct PointwiseBuf<'a> {
    builder: &'a PointwiseBuilder,
    node: NodeId,
}

impl<'a> PointwiseBuf<'a> {
    pub fn node(&self) -> NodeId {
        self.node
    }

    pub fn ty(&self) -> PType {
        self.builder.ir().ty(self.node).unwrap()
    }

    pub fn dtype(&self) -> DType {
        let PType::Pointer(dtype) = self.ty() else { panic!("Node is not a buffer!") };
        dtype
    }

    fn index(&self, idx: impl CoercesToPointwiseNode<'a>) -> NodeId {
        idx.coerce(self.builder, PType::Variable { ty: DType::I32, p2size: 0 }).node
    }

    pub fn read(self, idx: impl CoercesToPointwiseNode<'a>, p2size: u8) -> PointwiseNode<'a> {
        let idx = self.index(idx);
        let node = self.builder.add(|ir| ir.read(self.node, idx, p2size));
        PointwiseNode { builder: self.builder, node }
    }

    pub fn conditional_read(
        self,
        idx: impl CoercesToPointwiseNode<'a>,
        cond: impl CoercesToPointwiseNode<'a>,
        fallback: impl Into<DValue>,
        p2size: u8,
    ) -> PointwiseNode<'a> {
        let idx = self.index(idx);
        let cond = self.index(cond);
        let fallback = fallback.into();
        let node = self.builder.add(|ir| ir.conditional_read(self.node, idx, cond, fallback, p2size));
        PointwiseNode { builder: self.builder, node }
    }

    /// Store `val` at `idx`
    pub fn write(self, idx: impl CoercesToPointwiseNode<'a>, val: PointwiseNode<'a>) {
        let idx = self.index(idx);
        self.builder.ir().write(self.node, idx, val.node).unwrap();
    }

    /// Atomically accumulate `val` into `idx` - the buffer is zeroed before the
    /// kernel is launched
    pub fn atomic_add(self, idx: impl CoercesToPointwiseNode<'a>, val: PointwiseNode<'a>) {
        let idx = self.index(idx);
        self.builder.ir().atomic_add(self.node, idx, val.node).unwrap();
    }

    /// Apply `matmul(self, indices)`, where `self` holds the weights and `indices`
    /// the sparse input, and this thread produces the chunk of the output selected
    /// by its thread ID
    pub fn sparse_matmul(self, indices: Self, matmul: SparseMatmul, p2size: u8) -> PointwiseNode<'a> {
        let node = self.builder.add(|ir| ir.sparse_matmul(self.node, indices.node, p2size, matmul));
        PointwiseNode { builder: self.builder, node }
    }

    /// Backwards pass of [`PointwiseBuf::sparse_matmul`], accumulating into `self` -
    /// which is therefore zeroed before the kernel is launched
    pub fn sparse_matmul_bwd(self, indices: Self, gradients: PointwiseNode<'a>, matmul: SparseMatmul) {
        self.builder.ir().sparse_matmul_bwd(self.node, indices.node, gradients.node, matmul).unwrap();
    }
}

/// A value local to a single thread - `2^p2size` elements of a single dtype.
#[derive(Clone, Copy)]
pub struct PointwiseNode<'a> {
    builder: &'a PointwiseBuilder,
    node: NodeId,
}

impl<'a> PointwiseNode<'a> {
    pub fn node(&self) -> NodeId {
        self.node
    }

    pub fn ty(&self) -> PType {
        self.builder.ir().ty(self.node).unwrap()
    }

    pub fn dtype(&self) -> DType {
        let PType::Variable { ty, .. } = self.ty() else { panic!("Node is not a variable!") };
        ty
    }

    /// This node holds `2^p2size` elements
    pub fn p2size(&self) -> u8 {
        let PType::Variable { p2size, .. } = self.ty() else { panic!("Node is not a variable!") };
        p2size
    }

    /// Make a constant with the same dtype and vector width as `self`
    pub fn constant_like(self, value: impl Into<DValue>) -> Self {
        self.builder.new_constant(value, self.p2size())
    }

    /// Copy a scalar into each of the `2^p2size` elements
    pub fn broadcast(self, p2size: u8) -> Self {
        Self { node: self.builder.add(|ir| ir.broadcast(self.node, p2size)), ..self }
    }

    fn broadcast_to_same(self, rhs: Self) -> (Self, Self) {
        match (self.p2size(), rhs.p2size()) {
            (0, p2size) if p2size > 0 => (self.broadcast(p2size), rhs),
            (p2size, 0) if p2size > 0 => (self, rhs.broadcast(p2size)),
            _ => (self, rhs),
        }
    }

    /// Apply pointwise unary operation
    pub fn unary(self, unary: Unary) -> Self {
        Self { node: self.builder.add(|ir| ir.unary(self.node, unary)), ..self }
    }

    /// Apply pointwise binary operation
    pub fn binary(self, rhs: impl CoercesToPointwiseNode<'a>, binary: CABinary) -> Self {
        let rhs = rhs.coerce(self.builder, self.ty());
        let (lhs, rhs) = self.broadcast_to_same(rhs);
        Self { node: self.builder.add(|ir| ir.binary(lhs.node, rhs.node, binary)), ..self }
    }

    /// Apply `self^power`
    pub fn powf(self, power: impl CoercesToPointwiseNode<'a>) -> Self {
        let power = power.coerce(self.builder, self.ty());
        let (lhs, rhs) = self.broadcast_to_same(power);
        Self { node: self.builder.add(|ir| ir.powf(lhs.node, rhs.node)), ..self }
    }

    /// Apply `self / rhs` on scalar integers. Not the `Div` impl, as float
    /// division is instead done by multiplying by the reciprocal.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, rhs: impl CoercesToPointwiseNode<'a>) -> Self {
        let rhs = rhs.coerce(self.builder, self.ty());
        Self { node: self.builder.add(|ir| ir.div(self.node, rhs.node)), ..self }
    }

    /// Apply `self % rhs` on scalar integers
    #[allow(clippy::should_implement_trait)]
    pub fn rem(self, rhs: impl CoercesToPointwiseNode<'a>) -> Self {
        let rhs = rhs.coerce(self.builder, self.ty());
        Self { node: self.builder.add(|ir| ir.rem(self.node, rhs.node)), ..self }
    }

    /// Shorthand for `(self.div(rhs), self.rem(rhs))`, the usual way of splitting
    /// a thread ID into an index pair
    pub fn div_rem(self, rhs: impl CoercesToPointwiseNode<'a> + Copy) -> (Self, Self) {
        (self.div(rhs), self.rem(rhs))
    }

    pub fn min(self, value: impl CoercesToPointwiseNode<'a>) -> Self {
        self.binary(value, CABinary::Min)
    }

    pub fn max(self, value: impl CoercesToPointwiseNode<'a>) -> Self {
        self.binary(value, CABinary::Max)
    }

    pub fn abs(self) -> Self {
        self.unary(Unary::Abs)
    }

    pub fn sgn(self) -> Self {
        self.unary(Unary::Sgn)
    }

    pub fn exp(self) -> Self {
        self.unary(Unary::Exp)
    }

    pub fn log(self) -> Self {
        self.unary(Unary::Log)
    }

    pub fn sqrt(self) -> Self {
        self.unary(Unary::Sqrt)
    }

    pub fn recip(self) -> Self {
        self.unary(Unary::Reciprocal)
    }

    pub fn round(self) -> Self {
        self.unary(Unary::Round)
    }

    pub fn truncate(self) -> Self {
        self.unary(Unary::Truncate)
    }

    pub fn cast(self, dtype: DType) -> Self {
        self.unary(Unary::Cast(dtype))
    }

    /// Apply `self > 0`, as a `1` or `0` of the same dtype
    pub fn is_positive(self) -> Self {
        self.unary(Unary::IsPositive)
    }

    /// Apply `self >= 0`, as a `1` or `0` of the same dtype
    pub fn is_non_negative(self) -> Self {
        self.unary(Unary::IsNonNegative)
    }

    /// Apply `self == 0`, as a `1` or `0` of the same dtype
    pub fn is_zero(self) -> Self {
        self.unary(Unary::IsZero)
    }
}

impl<'a> Add<Self> for PointwiseNode<'a> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, CABinary::Add)
    }
}

impl<'a> Mul<Self> for PointwiseNode<'a> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, CABinary::Mul)
    }
}

impl<'a> Sub<Self> for PointwiseNode<'a> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Neg for PointwiseNode<'_> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.constant_like(DValue::neg_one(self.dtype())) * self
    }
}

impl Div<f32> for PointwiseNode<'_> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        (1.0 / rhs) * self
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> Div<PointwiseNode<'a>> for f32 {
    type Output = PointwiseNode<'a>;

    fn div(self, rhs: PointwiseNode<'a>) -> Self::Output {
        self * rhs.recip()
    }
}

macro_rules! impl_scalar_ops {
    ($($ty:ty),*) => {
        $(
            impl<'a> CoercesToPointwiseNode<'a> for $ty {
                fn coerce(&self, builder: &'a PointwiseBuilder, ty: PType) -> PointwiseNode<'a> {
                    builder.new_constant(*self, p2size_of(ty))
                }
            }

            impl<'a> Add<PointwiseNode<'a>> for $ty {
                type Output = PointwiseNode<'a>;

                fn add(self, rhs: PointwiseNode<'a>) -> Self::Output {
                    rhs.binary(self, CABinary::Add)
                }
            }

            impl Add<$ty> for PointwiseNode<'_> {
                type Output = Self;

                fn add(self, rhs: $ty) -> Self::Output {
                    self.binary(rhs, CABinary::Add)
                }
            }

            impl<'a> Mul<PointwiseNode<'a>> for $ty {
                type Output = PointwiseNode<'a>;

                fn mul(self, rhs: PointwiseNode<'a>) -> Self::Output {
                    rhs.binary(self, CABinary::Mul)
                }
            }

            impl Mul<$ty> for PointwiseNode<'_> {
                type Output = Self;

                fn mul(self, rhs: $ty) -> Self::Output {
                    self.binary(rhs, CABinary::Mul)
                }
            }

            impl<'a> Sub<PointwiseNode<'a>> for $ty {
                type Output = PointwiseNode<'a>;

                fn sub(self, rhs: PointwiseNode<'a>) -> Self::Output {
                    self + (-rhs)
                }
            }

            impl Sub<$ty> for PointwiseNode<'_> {
                type Output = Self;

                fn sub(self, rhs: $ty) -> Self::Output {
                    self + (-rhs)
                }
            }
        )*
    };
}

impl_scalar_ops!(f32, i32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmadd() {
        let ty = TType::new(4, DType::F32);

        let builder = PointwiseBuilder::new(4usize);
        let input1 = builder.new_buffer(ty);
        let input2 = builder.new_buffer(ty);

        let tid = builder.tid();
        let out = 2.0 * input1.read(tid, 2) + input2.read(tid, 2);
        input2.write(tid, out);

        builder.ir().eliminate_common_subexprs().unwrap();
        assert_eq!(builder.size().get(), 4);
        assert!(builder.inner().estimate_memory_cost().unwrap().get() > 0);
    }

    #[test]
    fn index_math_and_broadcast() {
        let builder = PointwiseBuilder::new(64usize);
        let buf = builder.new_buffer(TType::new(64, DType::F32));
        let idxs = builder.new_buffer(TType::new(64, DType::I32));

        let (outer, inner) = builder.tid().div_rem(8);
        let idx = 8 * outer + inner;

        // scalar read broadcast up to the kernel vector width
        let scale = buf.read(outer, 0).broadcast(1);
        let cond = (idx - 4).is_non_negative();
        let val = buf.conditional_read(idx, cond, 0.0f32, 1) * scale;
        let val = (val - 1.0).abs().powf(2.0).max(0.0);

        buf.atomic_add(idx, val);
        idxs.write(builder.tid(), idx.cast(DType::I32));

        builder.ir().eliminate_common_subexprs().unwrap();
    }
}
