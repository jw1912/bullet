use std::{
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::core::{DType, Size};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct IrType {
    size: Size,
    dtype: DType,
}

impl fmt::Debug for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}[{:?}]", self.dtype, self.size)
    }
}

impl IrType {
    pub fn new(size: impl Into<Size>, dtype: DType) -> Self {
        Self { size: size.into(), dtype }
    }

    pub fn size(&self) -> Size {
        self.size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct IrNodeId(usize);

impl IrNodeId {
    pub(super) fn new(id: usize) -> Self {
        Self(id)
    }
}

impl Default for IrNodeId {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Debug for IrNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IrNode {
    id: IrNodeId,
    ty: IrType,
    pub(super) children: usize,
}

impl IrNode {
    pub fn new(id: IrNodeId, ty: IrType) -> Self {
        Self { id, ty, children: 0 }
    }

    pub fn id(&self) -> IrNodeId {
        self.id
    }

    pub fn ty(&self) -> IrType {
        self.ty
    }

    pub fn children(&self) -> usize {
        self.children
    }

    pub fn inc_children(&mut self) {
        self.children += 1;
    }

    pub fn dec_children(&mut self) {
        self.children -= 1;
    }
}
