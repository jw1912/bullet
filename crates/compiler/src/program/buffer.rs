use std::{
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::ir::{node::DType, size::Size};

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct ProgramBufferId(usize);

impl Default for ProgramBufferId {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Debug for ProgramBufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

#[derive(Clone, Copy)]
pub struct ProgramBuffer {
    id: ProgramBufferId,
    dtype: DType,
    len: Size,
}

impl ProgramBuffer {
    pub fn new(dtype: DType, len: Size) -> Self {
        Self { id: ProgramBufferId::default(), dtype, len }
    }

    #[must_use]
    pub fn id(&self) -> ProgramBufferId {
        self.id
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn len(&self) -> Size {
        self.len
    }
}
