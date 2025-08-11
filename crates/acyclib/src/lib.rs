pub mod format;
pub mod topo;

use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Copy, Debug)]
pub enum GraphError {
    NodeDoesNotExist,
    NodeWithIdAlreadyExists,
    NodeIsSelfReferential,
    NodeIsNotRoot,
    Cyclic,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct NodeId(usize);

pub trait Operation: Clone {
    fn parents(&self) -> HashSet<NodeId>;
}

pub trait Type: Clone {}

#[derive(Clone)]
pub struct Node<Ty: Type, Op: Operation> {
    id: NodeId,
    ty: Ty,
    src: Option<Op>,
    children: usize,
}

impl<Ty: Type, Op: Operation> Node<Ty, Op> {
    fn new_node_id() -> NodeId {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        NodeId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub fn new(ty: Ty, src: Option<Op>) -> Self {
        Node { id: Self::new_node_id(), ty, src, children: 0 }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn ty(&self) -> Ty {
        self.ty.clone()
    }

    pub fn src(&self) -> &Option<Op> {
        &self.src
    }
}

pub struct Graph<Ty: Type, Op: Operation> {
    nodes: HashMap<NodeId, Node<Ty, Op>>,
}

impl<Ty: Type, Op: Operation> Default for Graph<Ty, Op> {
    fn default() -> Self {
        Self { nodes: HashMap::new() }
    }
}

impl<Ty: Type, Op: Operation> Graph<Ty, Op> {
    pub fn add_leaf(&mut self, ty: Ty) -> Result<NodeId, GraphError> {
        let node = Node::new(ty, None);
        let id = node.id;
        self.insert(node)?;
        Ok(id)
    }

    pub fn add_node(&mut self, ty: Ty, op: impl Into<Op>) -> Result<NodeId, GraphError> {
        let node = Node::new(ty, Some(op.into()));
        let id = node.id;
        self.insert(node)?;
        Ok(id)
    }

    pub fn get(&self, id: NodeId) -> Result<&Node<Ty, Op>, GraphError> {
        self.nodes.get(&id).ok_or(GraphError::NodeDoesNotExist)
    }

    fn get_mut(&mut self, id: NodeId) -> Result<&mut Node<Ty, Op>, GraphError> {
        self.nodes.get_mut(&id).ok_or(GraphError::NodeDoesNotExist)
    }

    pub fn roots(&self) -> HashSet<NodeId> {
        self.nodes.values().filter_map(|node| (node.children == 0).then_some(node.id)).collect()
    }

    pub fn topo_order(&self) -> Result<Vec<NodeId>, GraphError> {
        let edges_rev = self
            .nodes
            .iter()
            .map(|(&idx, data)| {
                let op = data.src.as_ref();
                let set = op.map(|x| x.parents().iter().map(|y| y.0).collect()).unwrap_or_default();
                (idx.0, set)
            })
            .collect();

        topo::topo_order(edges_rev).ok_or(GraphError::Cyclic).map(|x| x.into_iter().map(NodeId).collect())
    }

    pub fn insert(&mut self, node: Node<Ty, Op>) -> Result<(), GraphError> {
        if let Some(op) = node.src.as_ref() {
            for parent in op.parents() {
                if parent == node.id {
                    return Err(GraphError::NodeIsSelfReferential);
                }

                self.get_mut(parent)?.children += 1;
            }
        }

        self.nodes.insert(node.id, node).map_or(Ok(()), |_| Err(GraphError::NodeWithIdAlreadyExists))
    }

    pub fn remove(&mut self, id: NodeId) -> Result<(), GraphError> {
        let node = self.get(id)?;

        if node.children > 0 {
            return Err(GraphError::NodeIsNotRoot);
        }

        if let Some(op) = node.src.as_ref() {
            for parent in op.parents() {
                self.get_mut(parent)?.children -= 1;
            }
        }

        self.nodes.remove(&id).expect("Already verified node is present!");

        Ok(())
    }

    pub fn replace_op(&mut self, id: NodeId, new_op: impl Into<Op>) -> Result<(), GraphError> {
        let new_op = new_op.into();

        for parent in new_op.parents() {
            if parent == id {
                return Err(GraphError::NodeIsSelfReferential);
            }

            self.get_mut(parent)?.children += 1;
        }

        let node = self.get_mut(id)?;

        let mut new_src = Some(new_op);

        std::mem::swap(&mut new_src, &mut node.src);

        let old_src = new_src;

        if let Some(op) = old_src.as_ref() {
            for parent in op.parents() {
                self.get_mut(parent)?.children -= 1;
            }
        }

        Ok(())
    }

    pub fn eliminate_dead_nodes(&mut self, required: HashSet<NodeId>) -> Result<(), GraphError> {
        for id in self.topo_order()?.into_iter().rev() {
            if !required.contains(&id) && self.get(id)?.children == 0 {
                self.remove(id)?;
            }
        }

        Ok(())
    }
}
