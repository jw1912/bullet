use std::collections::HashMap;

use crate::{
    core::FormulaOp,
    ir::{
        IR, IRTrace,
        graph::operation::{FusedElementwise, IrOperation},
        transform::{EliminateUnusedOperations, IrTransform},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecomposeElementwise;

impl IrTransform for DecomposeElementwise {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        while decompose_single_elementwise(ir)? {}
        Ok(())
    }
}

fn decompose_single_elementwise(ir: &mut IR) -> Result<bool, IRTrace> {
    for op in ir.operations() {
        if let Some(elmt) = IrOperation::downcast::<FusedElementwise>(op.op()).cloned() {
            let mut values = elmt.input_ids().iter().zip(op.inputs()).map(|(&x, &y)| (x, y)).collect::<HashMap<_, _>>();

            let mut errored = false;
            elmt.desc().traverse(|id, op| {
                if errored {
                    return;
                }

                let get = |x| *values.get(&x).unwrap();

                match op {
                    FormulaOp::Leaf(_) => assert!(values.contains_key(&id)),
                    FormulaOp::Unary { input, op } => {
                        if let Ok(out) = ir.add_unary(get(input), op) {
                            errored |= values.insert(id, out).is_some();
                        } else {
                            errored = true;
                        }
                    }
                    FormulaOp::Binary { lhs, rhs, op } => {
                        if let Ok(out) = ir.add_binary(get(lhs), get(rhs), op) {
                            errored |= values.insert(id, out).is_some();
                        } else {
                            errored = true;
                        }
                    }
                }
            });

            if errored {
                return Err("Decompose_single_elementwise: error occurred in traversal!".into());
            }

            for (out, old) in elmt.output_ids().iter().zip(op.outputs().to_vec()) {
                let new = *values.get(out).ok_or("Decompose_single_elementwise: missing output value!")?;
                ir.swap_outputs(new, old)?;
            }

            ir.transform(EliminateUnusedOperations)?;

            return Ok(true);
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        core::{Binary, DType, Size},
        ir::IrType,
    };

    #[test]
    fn decompose_elementwise() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let ty = IrType::new(Size::variable(), DType::F32);
        let x = ir.add_leaf(ty);
        let y = ir.add_leaf(ty);
        let [z, w] = ir.add_elementwise([x, y], |builder, [x, y]| {
            let a = builder.binary(x, y, Binary::Add)?;
            let b = builder.binary_const(a, 1.0, Binary::Add, false)?;

            let c = builder.binary(x, y, Binary::Mul)?;
            let d = builder.binary_const(c, 1.0, Binary::Add, true)?;

            Some([b, d])
        })?;

        ir.register_output(z);
        ir.register_output(w);
        ir.transform(DecomposeElementwise)?;

        for op in ir.operations() {
            assert!(IrOperation::downcast::<FusedElementwise>(op.op()).is_none());
        }

        ir.check_valid()
    }
}
