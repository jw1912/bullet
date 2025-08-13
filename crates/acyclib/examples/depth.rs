use std::{collections::HashSet, fmt, rc::Rc};

use acyclib::graph::{Graph, GraphError, NodeId, Operation};

fn main() -> Result<(), GraphError> {
    let mut graph = Graph::<Ty, Op>::default();

    let a = graph.add_node(Leaf)?;
    let b = graph.add_node(Leaf)?;
    let c = graph.add_node(Leaf)?;
    let d = graph.add_node(Leaf)?;

    let x = graph.add_node(Mul(a, b))?;
    let y = graph.add_node(Mul(a, c))?;
    let z = graph.add_node(Add(a, x))?;
    let w = graph.add_node(Mul(d, d))?;

    let mut required = HashSet::new();
    required.insert(z);
    required.insert(y);

    println!("unused: {w:?}");

    println!("{}", graph.formatted()?);

    graph.replace_op(y, Add(a, c))?;

    println!("{}", graph.formatted()?);

    graph.eliminate_dead_nodes(required)?;

    println!("{}", graph.formatted()?);

    assert_eq!(graph.replace_op(w, Add(a, c)), Err(GraphError::NodeDoesNotExist));
    assert_eq!(graph.replace_op(z, Add(z, c)), Err(GraphError::FailedTypeCheck));

    Ok(())
}

trait Basic: fmt::Debug + 'static {
    fn par(&self) -> HashSet<NodeId>;
}

#[derive(Clone)]
struct Leaf;
impl Basic for Leaf {
    fn par(&self) -> HashSet<NodeId> {
        HashSet::new()
    }
}
impl fmt::Debug for Leaf {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

#[derive(Clone)]
struct Add(NodeId, NodeId);
impl Basic for Add {
    fn par(&self) -> HashSet<NodeId> {
        [self.0, self.1].into_iter().collect()
    }
}
impl fmt::Debug for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Add")
    }
}

#[derive(Clone)]
struct Mul(NodeId, NodeId);
impl Basic for Mul {
    fn par(&self) -> HashSet<NodeId> {
        [self.0, self.1].into_iter().collect()
    }
}
impl fmt::Debug for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mul")
    }
}

#[derive(Clone)]
struct Op(Rc<dyn Basic>);
impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: Basic> From<T> for Op {
    fn from(value: T) -> Self {
        Self(Rc::new(value))
    }
}

impl Operation<Ty> for Op {
    fn parents(&self) -> HashSet<NodeId> {
        self.0.par()
    }

    fn out_type(&self, graph: &Graph<Ty, Self>) -> Result<Ty, GraphError> {
        let mut ty = 0;

        for parent in self.0.par() {
            ty = ty.max(graph.get(parent)?.ty().0);
        }

        Ok(Ty(1 + ty))
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Ty(usize);
