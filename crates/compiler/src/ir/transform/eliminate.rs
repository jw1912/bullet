use crate::ir::{IR, IRTrace, operation::CopyOp, transform::IRTransform};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EliminateUnusedOperations;

impl IRTransform for EliminateUnusedOperations {
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

impl IRTransform for EliminateCopies {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        for op in ir.ordered_operations()? {
            let op = ir.get_op(op.id())?.clone();
            if op.downcast::<CopyOp>().is_some()
                && let [input] = op.inputs()[..]
                && let [output] = op.outputs()[..]
            {
                if !ir.is_output(output) {
                    ir.replace_input(input, output)?;
                    ir.remove_op(op.id())?;
                    continue;
                }

                if !ir.is_output(input) && !ir.is_input(input)? {
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

impl IRTransform for EliminateCommonSubExpressions {
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
    use crate::ir::{
        graph::{DType, TType, TValue},
        operation::CABinary,
    };

    use super::*;

    #[test]
    fn eliminate_unused_ops() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let x = ir.add_const(TValue::F32(vec![1.0; 8]));
        let y = ir.add_const(TValue::F32(vec![1.0; 8]));
        let z = ir.add_binary(x, y, CABinary::Add)?;
        let w = ir.add_binary(z, y, CABinary::Add)?;
        let t = ir.add_binary(w, y, CABinary::Add)?;

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
    fn eliminate_input_copy() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let ty = TType::new(1, DType::F32);
        let x = ir.add_input(ty);
        let y = ir.copy(x)?;
        let z = ir.copy(y)?;

        ir.register_output(z);
        ir.transform(EliminateCopies)?;

        assert_eq!(ir.num_ops(), 2);
        assert!(ir.get_node(x).is_ok());
        assert!(ir.get_node(z).is_ok());

        let op = ir.get_op(ir.get_parent_op(z)?)?;
        assert_eq!(op.inputs(), [x]);
        assert_eq!(op.outputs(), [z]);
        assert_eq!(op.downcast(), Some(&CopyOp(ty)));

        ir.check_valid()
    }

    #[test]
    fn eliminate_copies() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let x = ir.add_input(TType::new(1, DType::F32));
        let y = ir.copy(x)?;
        let z = ir.copy(y)?;
        let w = ir.copy(z)?;

        let a = ir.add_binary(x, y, CABinary::Add)?;
        let b = ir.add_binary(z, w, CABinary::Mul)?;
        let c = ir.add_binary(a, b, CABinary::Add)?;

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

        let x = ir.add_const(TValue::F32(vec![1.0; 8]));
        let y = ir.add_const(TValue::F32(vec![1.0; 8]));

        let z1 = ir.add_binary(x, y, CABinary::Add)?;
        let w1 = ir.add_binary(z1, y, CABinary::Add)?;
        let t1 = ir.add_binary(w1, y, CABinary::Mul)?;

        let z2 = ir.add_binary(x, y, CABinary::Add)?;
        let w2 = ir.add_binary(z2, y, CABinary::Add)?;
        let t2 = ir.add_binary(w2, y, CABinary::Mul)?;

        ir.register_output(t1);
        ir.register_output(t2);
        ir.register_output(x);
        ir.register_output(y);
        ir.transform(EliminateCommonSubExpressions)?;

        assert!(ir.are_copies(x, y)?);
        assert!(ir.are_copies(t1, t2)?);

        ir.check_valid()
    }
}
