use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

use crate::{
    common::{DType, DTypeTensor},
    ir::{
        IrError, IrGraph,
        node::IrNodeId,
        operation::{IrOperation, Leaf},
    },
};

impl IrGraph {
    pub fn evaluate(
        &self,
        inputs: impl Into<HashMap<IrNodeId, DTypeTensor>>,
    ) -> Result<HashMap<IrNodeId, DTypeTensor>, IrError> {
        let mut values: HashMap<_, _> =
            inputs.into().into_iter().map(|(id, tensor)| (id, RefCell::new(tensor))).collect();

        let mut vars = HashSet::new();

        for (id, tensor) in &values {
            let op = self.get_op(self.get_parent_op(*id)?)?;
            if IrOperation::downcast::<Leaf>(op.op()).is_none() {
                return Err("Seeded non-leaf node!".into());
            }

            let concrete_size = tensor.borrow().size();
            let size = self.get_node_type(*id)?.size();

            if let Some(var) = size.get_var_size(concrete_size) {
                vars.insert(var);
            }
        }

        let var = match vars.len() {
            0 => 1,
            1 => *vars.iter().next().unwrap(),
            _ => return Err(format!("Mismatching batch sizes in inputs: {vars:?}").into()),
        };

        for id in self.topo_order_ops()? {
            let op = self.get_op(id)?;

            for &output in op.outputs() {
                let ty = self.get_node_type(output)?;
                let size = ty.size().evaluate(var);

                let tensor = match ty.dtype() {
                    DType::F32 => DTypeTensor::F32(vec![0.0; size]),
                    DType::I32 => DTypeTensor::I32(vec![0; size]),
                };

                let is_prev = values.contains_key(&output);
                let is_leaf = IrOperation::downcast::<Leaf>(op.op()).is_some();

                if !is_leaf {
                    assert!(values.insert(output, RefCell::new(tensor)).is_none(), "Cannot happen!");
                } else if !is_prev {
                    return Err("Leaf node not seeded!".into());
                }
            }

            let op_inputs = op
                .inputs()
                .iter()
                .map(|i| values.get(i).map(|i| i.borrow()))
                .collect::<Option<Vec<_>>>()
                .ok_or("IrGraph::evaluate: input missing!")?;

            let mut op_outputs = op
                .outputs()
                .iter()
                .map(|i| values.get(i).map(|i| i.borrow_mut()))
                .collect::<Option<Vec<_>>>()
                .ok_or("IrGraph::evaluate: output missing!")?;

            op.op().evaluate(
                &op_inputs.iter().map(|x| &**x).collect::<Vec<_>>(),
                &mut op_outputs.iter_mut().map(|x| &mut **x).collect::<Vec<_>>(),
            );
        }

        Ok(values.into_iter().filter_map(|x| self.outputs.contains(&x.0).then(|| (x.0, x.1.into_inner()))).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        common::{Binary, Size},
        ir::node::IrType,
    };

    use super::*;

    #[test]
    fn evaluate() -> Result<(), IrError> {
        let mut ir = IrGraph::default();
        let size = Size::variable() * 2;

        let x = ir.add_leaf(IrType::new(size, DType::F32));
        let y = ir.add_leaf(IrType::new(size, DType::F32));
        let z = ir.add_leaf(IrType::new(size, DType::F32));

        let w = ir.add_binary(x, y, Binary::Add)?;
        let t = ir.add_binary(z, w, Binary::Mul)?;
        let u = ir.add_binary(t, x, Binary::Add)?;

        let ix = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);
        let iy = DTypeTensor::F32(vec![1.0, 1.0, 1.0, 1.0]);
        let iz = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);

        ir.register_output(u);

        let outputs = ir.evaluate([(x, ix), (y, iy), (z, iz)])?;

        assert_eq!(outputs, [(u, DTypeTensor::F32(vec![3.0, 8.0, 15.0, 24.0]))].into());

        ir.check_valid()
    }

    #[test]
    fn evaluate_missing_input() -> Result<(), IrError> {
        let mut ir = IrGraph::default();
        let size = Size::variable() * 2;

        let x = ir.add_leaf(IrType::new(size, DType::F32));
        let y = ir.add_leaf(IrType::new(size, DType::F32));
        let z = ir.add_binary(x, y, Binary::Add)?;
        ir.register_output(z);

        let ix = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ir.evaluate([(x, ix)]), Err("Leaf node not seeded!".into()));

        ir.check_valid()
    }

    #[test]
    fn evaluate_seed_non_leaf() -> Result<(), IrError> {
        let mut ir = IrGraph::default();
        let size = Size::variable() * 2;

        let x = ir.add_leaf(IrType::new(size, DType::F32));
        let y = ir.add_leaf(IrType::new(size, DType::F32));
        let z = ir.add_binary(x, y, Binary::Add)?;
        ir.register_output(z);

        let ix = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);
        let inputs = [(x, ix.clone()), (y, ix.clone()), (z, ix)];
        assert_eq!(ir.evaluate(inputs), Err("Seeded non-leaf node!".into()));

        ir.check_valid()
    }
}
