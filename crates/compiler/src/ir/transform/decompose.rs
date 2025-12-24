use std::collections::HashMap;

use crate::{
    elementwise::{Input, Operation},
    ir::{
        IrError, IrGraph,
        operation::{BroadcastAcrossDimension, IrElementwise, IrOperation},
    },
};

impl IrGraph {
    pub fn decompose_elementwise(&mut self) -> Result<(), IrError> {
        while self.decompose_single_elementwise()? {}
        Ok(())
    }

    fn decompose_single_elementwise(&mut self) -> Result<bool, IrError> {
        for op in self.ops.values().cloned().collect::<Vec<_>>() {
            if let Some(elmt) = IrOperation::downcast::<IrElementwise>(op.op()).cloned() {
                let mut values =
                    elmt.input_ids().iter().zip(op.inputs()).map(|(&x, &y)| (x, y)).collect::<HashMap<_, _>>();

                let mut errored = false;
                elmt.desc().traverse(|id, op| {
                    if errored {
                        return;
                    }

                    let mut map_it = |x| match x {
                        Input::Constant(val) => {
                            let constant = self.add_const(val.into());
                            let op = BroadcastAcrossDimension::new(val.dtype(), [1], 0, elmt.size());
                            self.add_op([constant], op.unwrap()).unwrap()[0]
                        }
                        Input::Index(x) => *values.get(&x).unwrap(),
                    };

                    match op {
                        Operation::Leaf(_) => assert!(values.contains_key(&id)),
                        Operation::Unary { input, op } => {
                            let input = map_it(input);
                            if let Ok(out) = self.add_unary(input, op) {
                                errored |= values.insert(id, out).is_some();
                            } else {
                                errored = true;
                            }
                        }
                        Operation::Binary { lhs, rhs, op } => {
                            let lhs = map_it(lhs);
                            let rhs = map_it(rhs);

                            if let Ok(out) = self.add_binary(lhs, rhs, op) {
                                errored |= values.insert(id, out).is_some();
                            } else {
                                errored = true;
                            }
                        }
                    }
                });

                if errored {
                    return Err("IrGraph::decompose_single_elementwise: error occurred in traversal!".into());
                }

                for (out, old) in elmt.output_ids().iter().zip(op.outputs().to_vec()) {
                    let new = *values.get(out).ok_or("IrGraph::decompose_single_elementwise: missing output value!")?;
                    self.swap_outputs(new, old)?;
                }

                self.eliminate_unused_ops()?;

                return Ok(true);
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        common::{DType, Size},
        ir::{IrError, IrGraph, node::IrType},
    };

    #[test]
    fn decompose_elementwise() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let ty = IrType::new(Size::variable(), DType::F32);
        let x = ir.add_leaf(ty);
        let y = ir.add_leaf(ty);
        let [z, w] = ir.add_elementwise([x, y], |[x, y]| Some([x + y + 1.0, x * y + 1.0]))?;

        ir.register_output(z);
        ir.register_output(w);
        ir.decompose_elementwise()?;
        ir.check_valid()
    }
}
