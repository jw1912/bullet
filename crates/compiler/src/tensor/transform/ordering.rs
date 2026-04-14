use std::{cmp::Ordering, collections::HashSet};

use crate::{
    ir::NodeId,
    tensor::{
        IRTrace, TensorIR,
        operation::{Constant, ScalarConstant},
        transform::IRTransform,
    },
};

fn rank_node(ir: &TensorIR, node: NodeId) -> Result<usize, IRTrace> {
    Ok(if ir.parent_op::<ScalarConstant>(node)?.is_some() {
        0
    } else if ir.parent_op::<Constant>(node)?.is_some() {
        1
    } else {
        2
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct NodeScore(NodeId, usize);

impl NodeScore {
    pub fn id(&self) -> NodeId {
        self.0
    }

    pub fn new(ir: &TensorIR, node: NodeId) -> Result<Self, IRTrace> {
        rank_node(ir, node).map(|rank| Self(node, rank))
    }
}

impl PartialOrd for NodeScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeScore {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.1.cmp(&other.1) {
            Ordering::Equal => self.0.cmp(&other.0),
            Ordering::Greater => Ordering::Greater,
            Ordering::Less => Ordering::Less,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderCommutativeInputs;

impl IRTransform for OrderCommutativeInputs {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        for op in ir.operations() {
            let groups = op.data().0.commutating_groups();

            for (i, group_i) in groups.iter().enumerate() {
                for group_j in groups.iter().skip(i + 1) {
                    if group_i.intersection(group_j).next().is_some() {
                        return Err("Distinct commutating groups intersect!".into());
                    }
                }
            }

            for group in groups {
                let mut group = group.into_iter().collect::<Vec<_>>();

                let mut nodes = Vec::with_capacity(group.len());
                for &i in &group {
                    let id = op.inputs()[i];
                    nodes.push(NodeScore::new(ir, id)?);
                }

                if op.data().0.inputs().iter().collect::<HashSet<_>>().len() > 1 {
                    return Err("Inputs within commutating group have differing types!".into());
                }

                group.sort();
                nodes.sort();

                let op = ir.get_op_mut(op.id())?;
                for (idx, NodeScore(id, _)) in group.into_iter().zip(nodes) {
                    op.set_input(idx, id);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tensor::{DType, TType, operation::CABinary};

    #[test]
    fn commutative_inputs() -> Result<(), IRTrace> {
        let mut ir = TensorIR::default();

        let ty = TType::new(1, DType::F32);
        let x = ir.add_input(ty);
        let y = ir.add_scalar(1.0, 1);
        let z = ir.add_binary(x, y, CABinary::Add)?;
        let w = ir.add_binary(y, x, CABinary::Add)?;
        let t = ir.add_binary(w, z, CABinary::Mul)?;

        ir.register_output(t);

        assert_eq!(ir.get_op(ir.get_parent_op(z)?)?.inputs(), &[x, y]);
        assert_eq!(ir.get_op(ir.get_parent_op(w)?)?.inputs(), &[y, x]);
        assert_eq!(ir.get_op(ir.get_parent_op(t)?)?.inputs(), &[w, z]);

        ir.transform(OrderCommutativeInputs)?;

        assert_eq!(ir.get_op(ir.get_parent_op(z)?)?.inputs(), &[y, x]);
        assert_eq!(ir.get_op(ir.get_parent_op(w)?)?.inputs(), &[y, x]);
        assert_eq!(ir.get_op(ir.get_parent_op(t)?)?.inputs(), &[z, w]);

        ir.check_valid()
    }
}
