use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    core::{Binary, DType, DTypeValue, Unary},
    utils::topo_order,
};

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct FormulaId(usize);

impl Default for FormulaId {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Debug for FormulaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, ".{:?}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FormulaOp {
    IrInput(DType),
    Unary { input: FormulaId, op: Unary },
    Binary { lhs: FormulaId, rhs: FormulaId, op: Binary },
}

impl FormulaOp {
    fn inputs(&self) -> Vec<FormulaId> {
        match self {
            Self::IrInput(_) => Vec::new(),
            Self::Unary { input, .. } => vec![*input],
            Self::Binary { lhs, rhs, .. } => vec![*lhs, *rhs],
        }
    }

    pub fn replace(&mut self, curr: FormulaId, new: FormulaId) {
        match self {
            Self::IrInput(_) => {}
            Self::Unary { input, .. } => {
                if *input == curr {
                    *input = new;
                }
            }
            Self::Binary { lhs, rhs, .. } => {
                if *lhs == curr {
                    *lhs = new;
                }

                if *rhs == curr {
                    *rhs = new;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Node {
    op: FormulaOp,
    ty: DType,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Formula {
    nodes: HashMap<FormulaId, Node>,
    locked: bool,
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let topo = self.topo_order();

        for x in topo {
            let op = self.nodes.get(&x).unwrap().op;
            writeln!(f, "{x:?} = {op:?}")?;
        }

        Ok(())
    }
}

impl Formula {
    pub fn evaluate(
        &self,
        mut values: HashMap<FormulaId, DTypeValue>,
        outputs: impl AsRef<[FormulaId]>,
    ) -> Option<Vec<DTypeValue>> {
        for id in self.topo_order() {
            let alr = values.contains_key(&id);
            let node = self.nodes.get(&id)?;
            let get = |input| values.get(&input).cloned();

            let value = match (node.op, alr) {
                (FormulaOp::IrInput(ty), true) => {
                    if ty == get(id)?.dtype() {
                        continue;
                    } else {
                        None
                    }
                }
                (FormulaOp::Unary { input, op }, false) => op.evaluate(get(input)?),
                (FormulaOp::Binary { lhs, rhs, op }, false) => op.evaluate(get(lhs)?, get(rhs)?),
                _ => None,
            }?;

            if values.insert(id, value).is_some() {
                return None;
            }
        }

        outputs.as_ref().iter().map(|id| values.get(id).cloned()).collect()
    }

    pub fn traverse(&self, mut f: impl FnMut(FormulaId, FormulaOp)) {
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

    pub fn num_children(&self, id: FormulaId) -> usize {
        let mut cnt = 0;

        for node in self.nodes.values() {
            if node.op.inputs().contains(&id) {
                cnt += 1;
            }
        }

        cnt
    }

    pub fn is_root(&self, id: FormulaId) -> bool {
        self.num_children(id) == 0
    }

    pub fn num_parents(&self, id: FormulaId) -> usize {
        self.nodes.get(&id).unwrap().op.inputs().len()
    }

    pub fn is_leaf(&self, id: FormulaId) -> bool {
        self.num_parents(id) == 0
    }

    pub fn roots(&self) -> usize {
        self.nodes.keys().filter(|&&x| self.is_root(x)).count()
    }

    pub fn leaves(&self) -> usize {
        self.nodes.keys().filter(|&&x| self.is_leaf(x)).count()
    }

    pub fn topo_order(&self) -> Vec<FormulaId> {
        let edges_rev =
            self.nodes.iter().map(|(&idx, data)| (idx.0, data.op.inputs().iter().map(|x| x.0).collect())).collect();

        topo_order(edges_rev).unwrap().iter().map(|&x| FormulaId(x)).collect()
    }

    pub fn get_dtype(&self, input: FormulaId) -> DType {
        self.nodes.get(&input).unwrap().ty
    }

    fn add_op(&mut self, op: FormulaOp) -> Option<FormulaId> {
        let output = FormulaId::default();

        let ty = match &op {
            FormulaOp::IrInput(ty) => *ty,
            FormulaOp::Unary { input, op } => op.dtype(self.get_dtype(*input))?,
            FormulaOp::Binary { lhs, rhs, op } => op.dtype(self.get_dtype(*lhs), self.get_dtype(*rhs))?,
        };

        let node = Node { op, ty };
        self.nodes.insert(output, node);

        Some(output)
    }

    pub fn lock_inputs(&mut self) {
        self.locked = true;
    }

    pub fn input(&mut self, dtype: DType) -> Option<FormulaId> {
        if self.locked {
            return None;
        }

        self.add_op(FormulaOp::IrInput(dtype))
    }

    pub fn unary(&mut self, input: FormulaId, op: Unary) -> Option<FormulaId> {
        self.add_op(FormulaOp::Unary { input, op })
    }

    pub fn binary(&mut self, lhs: FormulaId, rhs: FormulaId, op: Binary) -> Option<FormulaId> {
        self.add_op(FormulaOp::Binary { lhs, rhs, op })
    }

    pub fn binary_const(
        &mut self,
        input: FormulaId,
        val: impl Into<DTypeValue>,
        op: Binary,
        lhs: bool,
    ) -> Option<FormulaId> {
        self.add_op(FormulaOp::Unary { input, op: Unary::BinaryWithConst { op, val: val.into(), lhs } })
    }

    pub fn merge_with(&self, rhs: &Self, equivalencies: &[(FormulaId, FormulaId)]) -> Option<Self> {
        let mut res = self.clone();

        for (&id, &node) in rhs.nodes.iter() {
            res.nodes.insert(id, node);
        }

        for &(a, b) in equivalencies {
            let &Node { op: op_a, ty } = res.nodes.get(&a).unwrap();
            let &Node { op: op_b, ty: ty_b } = res.nodes.get(&b).unwrap();

            assert_eq!(ty, ty_b);

            if let FormulaOp::IrInput(_) = op_a {
                res.nodes.get_mut(&a).unwrap().op = op_b;
            } else if let FormulaOp::IrInput(_) = op_b {
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

    pub fn relabel(&mut self, relabels: &[(FormulaId, FormulaId)]) -> Option<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_elementwise() {
        let mut elmt = Formula::default();
        let a = elmt.input(DType::F32).unwrap();
        let b = elmt.input(DType::F32).unwrap();

        let c = elmt.binary(a, b, Binary::Add).unwrap();

        assert_eq!(elmt.len(), 3);
        assert_eq!(elmt.roots(), 1);
        assert_eq!(elmt.leaves(), 2);
        assert!(elmt.is_root(c));
    }

    #[test]
    fn merge() {
        let mut elmt1 = Formula::default();
        let a = elmt1.input(DType::F32).unwrap();
        let b = elmt1.input(DType::F32).unwrap();

        let c = elmt1.binary(a, b, Binary::Add).unwrap();

        let mut elmt2 = Formula::default();
        let c2 = elmt2.input(DType::F32).unwrap();
        let d = elmt2.input(DType::F32).unwrap();

        let e = elmt2.binary(c2, d, Binary::Add).unwrap();

        let elmt = elmt1.merge_with(&elmt2, &[(c, c2)]).unwrap();

        assert_eq!(elmt.len(), 5);
        assert_eq!(elmt.roots(), 1);
        assert_eq!(elmt.leaves(), 3);
        assert!(elmt.is_root(e));

        let mut expected = Formula::default();
        let exp_a = expected.input(DType::F32).unwrap();
        let exp_b = expected.input(DType::F32).unwrap();
        let exp_c = expected.binary(exp_a, exp_b, Binary::Add).unwrap();
        let exp_d = expected.input(DType::F32).unwrap();
        let exp_e = expected.binary(exp_c, exp_d, Binary::Add).unwrap();

        expected.relabel(&[(exp_a, a), (exp_b, b), (exp_c, c), (exp_d, d), (exp_e, e)]).unwrap();

        assert_eq!(elmt, expected);
    }

    #[test]
    fn invalid_merge() {
        let mut elmt1 = Formula::default();
        let a = elmt1.input(DType::F32).unwrap();
        let b = elmt1.input(DType::F32).unwrap();

        let c = elmt1.binary(a, b, Binary::Add).unwrap();

        let mut elmt2 = Formula::default();
        let c2 = elmt2.input(DType::F32).unwrap();
        let d = elmt2.input(DType::F32).unwrap();

        let e = elmt2.binary(c2, d, Binary::Add).unwrap();

        assert!(elmt1.merge_with(&elmt2, &[(c, e)]).is_none());
    }

    #[test]
    fn evaluate() {
        let mut elmt = Formula::default();
        let fp_a = elmt.input(DType::F32).unwrap();
        let fp_b = elmt.input(DType::F32).unwrap();

        let fp_c = elmt.binary(fp_a, fp_b, Binary::Add).unwrap();

        let int_a = elmt.input(DType::I32).unwrap();
        let int_b = elmt.input(DType::I32).unwrap();

        let int_c = elmt.binary(int_a, int_b, Binary::Add).unwrap();

        let fp_int_c = elmt.unary(int_c, Unary::Cast(DType::F32)).unwrap();

        let out = elmt.binary(fp_c, fp_int_c, Binary::Div).unwrap();

        let inputs = [
            (fp_a, DTypeValue::F32(1.0)),
            (fp_b, DTypeValue::F32(2.0)),
            (int_a, DTypeValue::I32(1)),
            (int_b, DTypeValue::I32(1)),
        ]
        .into();
        let values = elmt.evaluate(inputs, [out]).unwrap();

        assert_eq!(values.len(), 1);
        assert_eq!(values[0], DTypeValue::F32(1.5));
    }

    #[test]
    fn evaluate_invalid_input() {
        let mut elmt = Formula::default();
        let a = elmt.input(DType::F32).unwrap();
        let b = elmt.input(DType::F32).unwrap();

        let c = elmt.binary(a, b, Binary::Add).unwrap();

        let inputs = [(a, DTypeValue::F32(1.0)), (b, DTypeValue::I32(1))].into();
        assert!(elmt.evaluate(inputs, [c]).is_none());
    }
}
