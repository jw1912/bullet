use std::collections::HashSet;

use crate::ir::{IrError, IrGraph};

impl IrGraph {
    /// Put commuting inputs to operations into canoncial order:
    /// ```text
    /// irgraph(%0: f32[1], %1: f32[1]) {
    ///     %2 = %0 + %1
    ///     %3 = %1 + %0
    ///     %4 = %3 * %2
    ///     return %4
    /// }
    /// ```
    /// Is rewritten to
    /// ```text
    /// irgraph(%0: f32[1], %1: f32[1]) {
    ///     %2 = %0 + %1
    ///     %3 = %0 + %1
    ///     %4 = %2 * %3
    ///     return %4
    /// }
    /// ```
    /// This makes techniques such as common subexpression
    /// elimination easier to perform
    pub fn canonicalise_inputs(&mut self) -> Result<(), IrError> {
        for op_id in self.topo_order_ops()? {
            let op = self.get_op(op_id)?;
            let groups = op.op().commutating_groups();
            let inputs = op.inputs().to_vec();

            for (i, group_i) in groups.iter().enumerate() {
                for group_j in groups.iter().skip(i + 1) {
                    if group_i.intersection(group_j).next().is_some() {
                        return Err("IrGraph::canonicalise: distinct commutating groups intersect!".into());
                    }
                }
            }

            for group in groups {
                let mut group = group.into_iter().collect::<Vec<_>>();
                let mut nodes = group.iter().map(|&i| inputs[i]).collect::<Vec<_>>();

                if nodes.iter().map(|&id| self.get_node_type(id)).collect::<Result<HashSet<_>, _>>()?.len() > 1 {
                    return Err("IrGraph::canonicalise: inputs within commutating group have differing types!".into());
                }

                group.sort();
                nodes.sort();

                for (idx, id) in group.into_iter().zip(nodes) {
                    self.get_op_mut(op_id)?.set_input(idx, id);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        common::{Binary, DType},
        ir::{IrError, IrGraph, node::IrType},
    };

    #[test]
    fn inputs() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let ty = IrType::new(1, DType::F32);
        let x = ir.add_leaf(ty);
        let y = ir.add_leaf(ty);
        let z = ir.add_binary(x, y, Binary::Add)?;
        let w = ir.add_binary(y, x, Binary::Add)?;
        let t = ir.add_binary(w, z, Binary::Mul)?;

        ir.register_output(t);

        assert_eq!(ir.get_op(ir.get_parent_op(z)?)?.inputs(), &[x, y]);
        assert_eq!(ir.get_op(ir.get_parent_op(w)?)?.inputs(), &[y, x]);
        assert_eq!(ir.get_op(ir.get_parent_op(t)?)?.inputs(), &[w, z]);

        ir.canonicalise_inputs()?;

        assert_eq!(ir.get_op(ir.get_parent_op(z)?)?.inputs(), &[x, y]);
        assert_eq!(ir.get_op(ir.get_parent_op(w)?)?.inputs(), &[x, y]);
        assert_eq!(ir.get_op(ir.get_parent_op(t)?)?.inputs(), &[z, w]);

        ir.check_valid()
    }
}
