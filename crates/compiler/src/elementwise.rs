mod builder;
mod kernel;

#[cfg(test)]
mod tests;

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
};

pub use builder::{ElementwiseBuilder, ElementwiseNode};
pub use kernel::{ElementwiseKernel, ElementwiseKernelBuilder, ElementwiseMut, ElementwiseRef};

use crate::common::{Binary, DType, DTypeValue, Unary, topo_order};

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct ElementwiseId(usize);

impl Default for ElementwiseId {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Debug for ElementwiseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, ".{:?}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Input {
    Index(ElementwiseId),
    Constant(DTypeValue),
}

impl Input {
    pub fn replace(&mut self, curr: ElementwiseId, new: ElementwiseId) {
        if let Input::Index(x) = *self
            && x == curr
        {
            *self = Input::Index(new);
        }
    }
}

impl From<ElementwiseId> for Input {
    fn from(value: ElementwiseId) -> Self {
        Input::Index(value)
    }
}

impl From<DTypeValue> for Input {
    fn from(value: DTypeValue) -> Self {
        Input::Constant(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operation {
    Leaf(DType),
    Unary { input: Input, op: Unary },
    Binary { lhs: Input, rhs: Input, op: Binary },
}

impl Operation {
    fn inputs(&self) -> Vec<ElementwiseId> {
        let mut res = Vec::new();

        match self {
            Self::Leaf(_) => {}
            Self::Unary { input, .. } => {
                if let Input::Index(x) = *input {
                    res.push(x)
                }
            }
            Self::Binary { lhs, rhs, .. } => {
                if let Input::Index(x) = *lhs {
                    res.push(x);
                }

                if let Input::Index(x) = *rhs {
                    res.push(x);
                }
            }
        }

        res
    }

    pub fn replace(&mut self, curr: ElementwiseId, new: ElementwiseId) {
        match self {
            Self::Leaf(_) => {}
            Self::Unary { input, .. } => input.replace(curr, new),
            Self::Binary { lhs, rhs, .. } => {
                lhs.replace(curr, new);
                rhs.replace(curr, new);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Node {
    op: Operation,
    ty: DType,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElementwiseDescription {
    nodes: HashMap<ElementwiseId, Node>,
}

impl fmt::Display for ElementwiseDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let topo = self.topo_order();

        for x in topo {
            let op = self.nodes.get(&x).unwrap().op;
            writeln!(f, "{x:?} = {op:?}")?;
        }

        Ok(())
    }
}

impl ElementwiseDescription {
    pub fn evaluate(
        &self,
        mut values: HashMap<ElementwiseId, DTypeValue>,
        outputs: impl AsRef<[ElementwiseId]>,
    ) -> Option<Vec<DTypeValue>> {
        for id in self.topo_order() {
            let alr = values.contains_key(&id);
            let node = self.nodes.get(&id)?;

            let get_input = |input| match input {
                Input::Constant(val) => Some(val),
                Input::Index(id) => values.get(&id).cloned(),
            };

            let value = match (node.op, alr) {
                (Operation::Leaf(ty), true) => {
                    if ty == values.get(&id).unwrap().dtype() {
                        continue;
                    } else {
                        None
                    }
                }
                (Operation::Unary { input, op }, false) => op.evaluate(get_input(input)?),
                (Operation::Binary { lhs, rhs, op }, false) => op.evaluate(get_input(lhs)?, get_input(rhs)?),
                _ => None,
            }?;

            assert!(values.insert(id, value).is_none(), "Have 'continue'd already!");
        }

        outputs.as_ref().iter().map(|id| values.get(id).cloned()).collect()
    }

    pub fn traverse(&self, mut f: impl FnMut(ElementwiseId, Operation)) {
        let topo = self.topo_order();

        for x in topo {
            f(x, self.nodes.get(&x).unwrap().op);
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_children(&self, id: ElementwiseId) -> usize {
        let mut cnt = 0;

        for node in self.nodes.values() {
            if node.op.inputs().contains(&id) {
                cnt += 1;
            }
        }

        cnt
    }

    pub fn is_root(&self, id: ElementwiseId) -> bool {
        self.num_children(id) == 0
    }

    pub fn num_parents(&self, id: ElementwiseId) -> usize {
        self.nodes.get(&id).unwrap().op.inputs().len()
    }

    pub fn is_leaf(&self, id: ElementwiseId) -> bool {
        self.num_parents(id) == 0
    }

    pub fn roots(&self) -> usize {
        self.nodes.keys().filter(|&&x| self.is_root(x)).count()
    }

    pub fn leaves(&self) -> usize {
        self.nodes.keys().filter(|&&x| self.is_leaf(x)).count()
    }

    pub fn topo_order(&self) -> Vec<ElementwiseId> {
        let edges_rev =
            self.nodes.iter().map(|(&idx, data)| (idx.0, data.op.inputs().iter().map(|x| x.0).collect())).collect();

        topo_order(edges_rev).unwrap().iter().map(|&x| ElementwiseId(x)).collect()
    }

    pub fn get_dtype(&self, input: impl Into<Input>) -> DType {
        match input.into() {
            Input::Constant(val) => val.dtype(),
            Input::Index(idx) => self.nodes.get(&idx).unwrap().ty,
        }
    }

    pub fn add_op(&mut self, op: Operation) -> Option<ElementwiseId> {
        let output = ElementwiseId::default();

        let ty = match &op {
            Operation::Leaf(ty) => *ty,
            Operation::Unary { input, op } => op.dtype(self.get_dtype(*input)),
            Operation::Binary { lhs, rhs, op } => op.dtype(self.get_dtype(*lhs), self.get_dtype(*rhs))?,
        };

        let node = Node { op, ty };
        self.nodes.insert(output, node);

        Some(output)
    }

    pub fn add_input(&mut self, dtype: DType) -> ElementwiseId {
        self.add_op(Operation::Leaf(dtype)).unwrap()
    }

    pub fn unary(&mut self, input: ElementwiseId, op: Unary) -> Option<ElementwiseId> {
        self.add_op(Operation::Unary { input: Input::Index(input), op })
    }

    pub fn binary(&mut self, lhs: impl Into<Input>, rhs: impl Into<Input>, op: Binary) -> Option<ElementwiseId> {
        let lhs = lhs.into();
        let rhs = rhs.into();

        self.add_op(Operation::Binary { lhs, rhs, op })
    }

    pub fn merge_with(&self, rhs: &Self, equivalencies: &[(ElementwiseId, ElementwiseId)]) -> Option<Self> {
        let mut res = self.clone();

        for (&id, &node) in rhs.nodes.iter() {
            res.nodes.insert(id, node);
        }

        for &(a, b) in equivalencies {
            let &Node { op: op_a, ty } = res.nodes.get(&a).unwrap();
            let &Node { op: op_b, ty: ty_b } = res.nodes.get(&b).unwrap();

            assert_eq!(ty, ty_b);

            if let Operation::Leaf(_) = op_a {
                res.nodes.get_mut(&a).unwrap().op = op_b;
            } else if let Operation::Leaf(_) = op_b {
            } else {
                return None;
            }

            res.nodes.remove(&b);
            for node in res.nodes.values_mut() {
                node.op.replace(b, a);
            }
        }

        Some(res)
    }

    pub fn relabel(&mut self, relabels: &[(ElementwiseId, ElementwiseId)]) -> Option<()> {
        let removals = relabels.iter().map(|x| x.0).collect::<HashSet<_>>();
        let insertions = relabels.iter().map(|x| x.1).collect::<HashSet<_>>();

        if !removals.is_disjoint(&insertions) {
            return None;
        }

        for &(a, b) in relabels {
            let node = self.nodes.remove(&a)?;
            self.nodes.insert(b, node);

            for node in self.nodes.values_mut() {
                node.op.replace(a, b);
            }
        }

        Some(())
    }
}
