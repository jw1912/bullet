use std::{collections::HashSet, fmt, rc::Rc};

use acyclib::{Graph, GraphError, NodeId, Operation, Type};

trait Binary: fmt::Debug {
    fn par(&self) -> (NodeId, NodeId);
}

#[derive(Clone)]
struct Add(NodeId, NodeId);
impl Binary for Add {
    fn par(&self) -> (NodeId, NodeId) {
        (self.0, self.1)
    }
}
impl fmt::Debug for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Add")
    }
}

#[derive(Clone)]
struct Mul(NodeId, NodeId);
impl Binary for Mul {
    fn par(&self) -> (NodeId, NodeId) {
        (self.0, self.1)
    }
}
impl fmt::Debug for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mul")
    }
}

#[derive(Clone)]
struct Op(Rc<dyn Binary>);
impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Op {
    fn new(inner: impl Binary + 'static) -> Self {
        Self(Rc::new(inner))
    }
}

impl Operation for Op {
    fn parents(&self) -> HashSet<NodeId> {
        let parents = self.0.par();
        let mut res = HashSet::with_capacity(2);
        res.insert(parents.0);
        res.insert(parents.1);
        res
    }
}

#[derive(Clone, Debug)]
struct Ty;

impl Type for Ty {}

fn main() -> Result<(), GraphError> {
    let mut graph = Graph::<Ty, Op>::default();

    let a = graph.add_leaf(Ty)?;
    let b = graph.add_leaf(Ty)?;
    let c = graph.add_leaf(Ty)?;
    let d = graph.add_leaf(Ty)?;

    let x = graph.add_node(Ty, Op::new(Mul(a, b)))?;
    let y = graph.add_node(Ty, Op::new(Mul(a, c)))?;
    let z = graph.add_node(Ty, Op::new(Add(a, x)))?;

    let mut required = HashSet::new();
    required.insert(z);

    println!("unused: {d:?}, {y:?}");

    println!("{graph}");

    graph.eliminate_dead_nodes(required)?;

    println!("{graph}");

    Ok(())
}
