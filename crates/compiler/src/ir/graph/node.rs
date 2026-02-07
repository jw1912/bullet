use std::{
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::ir::graph::TType;

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(usize);

impl NodeId {
    pub(super) fn new(id: usize) -> Self {
        Self(id)
    }
}

impl Default for NodeId {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Node {
    id: NodeId,
    ty: TType,
    pub(super) children: usize,
}

impl Node {
    pub fn new(id: NodeId, ty: TType) -> Self {
        Self { id, ty, children: 0 }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn ty(&self) -> TType {
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
