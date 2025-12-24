use std::collections::HashSet;

use crate::ir::{IrError, IrGraph};

impl IrGraph {
    pub fn canonicalise(&mut self) -> Result<(), IrError> {
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
        common::{Binary, DTypeTensor},
        ir::{IrError, IrGraph},
    };

    #[test]
    fn canonicalise() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let y = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let z = ir.add_binary(y, x, Binary::Add)?;

        ir.register_output(z);

        let op = ir.get_op(ir.get_parent_op(z)?)?;
        assert_eq!(op.inputs(), &[y, x]);

        ir.canonicalise()?;

        assert!(ir.get_node(x).is_ok());
        assert!(ir.get_node(y).is_ok());
        assert!(ir.get_node(z).is_ok());

        let op = ir.get_op(ir.get_parent_op(z)?)?;
        assert_eq!(op.inputs(), &[x, y]);

        ir.check_valid()
    }
}
