pub mod topo;

use std::{
    collections::{HashMap, HashSet},
    fmt,
    rc::Rc,
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

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

pub trait Operation: Clone + fmt::Debug {
    fn parents(&self) -> HashSet<NodeId>;
}

pub trait Type: Clone + fmt::Debug {}

#[derive(Clone)]
pub struct Node<Ty: Type, Op: Operation> {
    id: NodeId,
    ty: Ty,
    src: Option<Rc<Op>>,
    children: usize,
}

impl<Ty: Type, Op: Operation> Node<Ty, Op> {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn ty(&self) -> Ty {
        self.ty.clone()
    }

    pub fn src(&self) -> &Option<Rc<Op>> {
        &self.src
    }
}

pub struct Graph<Ty: Type, Op: Operation> {
    nodes: HashMap<NodeId, Node<Ty, Op>>,
    counter: AtomicUsize,
}

impl<Ty: Type, Op: Operation> Default for Graph<Ty, Op> {
    fn default() -> Self {
        Self { nodes: HashMap::new(), counter: AtomicUsize::new(0) }
    }
}

impl<Ty: Type, Op: Operation> fmt::Display for Graph<Ty, Op> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn map<T>(res: Result<T, GraphError>) -> Result<T, fmt::Error> {
            res.map_err(|_| fmt::Error)
        }

        let order = map(self.topo_order())?;

        write!(f, "## Graph ##")?;

        for id in order {
            writeln!(f)?;

            let node = map(self.get(id))?;
            let Node { id, ty, src, .. } = node;
            write!(f, "{id:?}: {ty:?}")?;

            if let Some(op) = src {
                write!(f, " = [{op:?}] {:?}", op.parents())?;
            }

            write!(f, ";")?;
        }

        Ok(())
    }
}

impl<Ty: Type, Op: Operation> Graph<Ty, Op> {
    fn new_node_id(&self) -> NodeId {
        NodeId(self.counter.fetch_add(1, Ordering::Relaxed))
    }

    pub fn new_leaf(&self, ty: Ty) -> Node<Ty, Op> {
        Node { id: self.new_node_id(), ty, src: None, children: 0 }
    }

    pub fn new_node(&self, ty: Ty, op: Op) -> Node<Ty, Op> {
        Node { id: self.new_node_id(), ty, src: Some(Rc::new(op)), children: 0 }
    }

    pub fn add_leaf(&mut self, ty: Ty) -> Result<NodeId, GraphError> {
        let node = self.new_leaf(ty);
        let id = node.id;
        self.insert(node)?;
        Ok(id)
    }

    pub fn add_node(&mut self, ty: Ty, op: Op) -> Result<NodeId, GraphError> {
        let node = self.new_node(ty, op);
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

    pub fn replace_op(&mut self, id: NodeId, new_op: Op) -> Result<(), GraphError> {
        for parent in new_op.parents() {
            if parent == id {
                return Err(GraphError::NodeIsSelfReferential);
            }

            self.get_mut(parent)?.children += 1;
        }

        let node = self.get_mut(id)?;

        let mut new_src = Some(Rc::new(new_op));

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
