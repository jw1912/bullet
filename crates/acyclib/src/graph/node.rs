use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct NodeId(pub(super) usize);

impl NodeId {
    pub fn inner(self) -> usize {
        self.0
    }
}

#[derive(Clone)]
pub struct Node<Ty: Clone, Op: Clone> {
    pub(super) id: NodeId,
    pub(super) ty: Ty,
    pub(super) op: Op,
    pub(super) children: usize,
}

impl<Ty: Clone, Op: Clone> Node<Ty, Op> {
    fn new_node_id() -> NodeId {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        NodeId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub(super) fn new(ty: Ty, op: impl Into<Op>) -> Self {
        Self { id: Self::new_node_id(), ty, op: op.into(), children: 0 }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn ty(&self) -> Ty {
        self.ty.clone()
    }

    pub fn op(&self) -> &Op {
        &self.op
    }

    pub fn children(&self) -> usize {
        self.children
    }
}
