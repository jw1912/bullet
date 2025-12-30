use crate::ir::{IR, IRTrace, graph::operation::IrCopy, transform::IrTransform};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EliminateUnusedOperations;

impl IrTransform for EliminateUnusedOperations {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        for op in ir.ordered_operations()?.into_iter().rev() {
            if op.outputs().iter().all(|&output| {
                let node = ir.get_node(output).unwrap();
                node.children() == 0 && !ir.is_output(output)
            }) {
                ir.remove_op(op.id())?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EliminateCopies;

impl IrTransform for EliminateCopies {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        for op in ir.ordered_operations()?.into_iter().rev() {
            if op.downcast::<IrCopy>().is_some()
                && let [input] = op.inputs()[..]
                && let [output] = op.outputs()[..]
            {
                if !ir.is_output(output) {
                    ir.replace_input(input, output)?;
                    ir.remove_op(op.id())?;
                    continue;
                }

                if !ir.is_output(input) {
                    // no topo check as performs y = copy(x) -> y = copy(y)
                    ir.graph.replace_input_unchecked(output, input)?;
                    ir.swap_outputs(input, output)?;
                    ir.remove_op(op.id())?;
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EliminateCommonSubExpressions;

impl IrTransform for EliminateCommonSubExpressions {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        while eliminate_single_common_subexpr(ir)? {}
        Ok(())
    }
}

fn eliminate_single_common_subexpr(ir: &mut IR) -> Result<bool, IRTrace> {
    let ops = ir.operations();

    for (i, op_i) in ops.iter().enumerate() {
        for op_j in ops.iter().skip(i + 1) {
            if op_i.inputs() == op_j.inputs() && op_i.op().equals(op_j.op()) {
                for (&out_i, &out_j) in op_i.outputs().iter().zip(op_j.outputs()) {
                    ir.replace_input(out_i, out_j)?;

                    if ir.is_output(out_j) {
                        let new_out = ir.copy(out_i)?;
                        ir.swap_outputs(new_out, out_j)?;
                    }
                }

                ir.remove_op(op_j.id())?;

                return Ok(true);
            }
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use crate::{
        core::{Binary, DType, DTypeTensor},
        ir::graph::IrType,
    };

    use super::*;

    #[test]
    fn eliminate_unused_ops() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let x = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let y = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let z = ir.add_binary(x, y, Binary::Add)?;
        let w = ir.add_binary(z, y, Binary::Add)?;
        let t = ir.add_binary(w, y, Binary::Add)?;

        ir.register_output(z);

        ir.transform(EliminateUnusedOperations)?;

        assert!(ir.get_node(x).is_ok());
        assert!(ir.get_node(y).is_ok());
        assert!(ir.get_node(z).is_ok());
        assert!(ir.get_node(w).is_err());
        assert!(ir.get_node(t).is_err());

        ir.check_valid()
    }

    #[test]
    fn eliminate_copies() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let x = ir.add_input(IrType::new(1, DType::F32));
        let y = ir.copy(x)?;
        let z = ir.copy(y)?;
        let w = ir.copy(z)?;

        let a = ir.add_binary(x, y, Binary::Add)?;
        let b = ir.add_binary(z, w, Binary::Mul)?;
        let c = ir.add_binary(a, b, Binary::Add)?;

        ir.register_output(a);
        ir.register_output(b);
        ir.register_output(c);
        ir.transform(EliminateCopies)?;

        assert_eq!(ir.num_ops(), 4);

        ir.check_valid()
    }

    #[test]
    fn eliminate_common_subexprs() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let x = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let y = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));

        let z1 = ir.add_binary(x, y, Binary::Add)?;
        let w1 = ir.add_binary(z1, y, Binary::Add)?;
        let t1 = ir.add_binary(w1, y, Binary::Mul)?;

        let z2 = ir.add_binary(x, y, Binary::Add)?;
        let w2 = ir.add_binary(z2, y, Binary::Add)?;
        let t2 = ir.add_binary(w2, y, Binary::Mul)?;

        ir.register_output(t1);
        ir.register_output(t2);
        ir.transform(EliminateCommonSubExpressions)?;

        assert!(ir.get_node(x).is_ok());
        assert!(ir.get_node(y).is_ok());

        ir.check_valid()
    }
}
