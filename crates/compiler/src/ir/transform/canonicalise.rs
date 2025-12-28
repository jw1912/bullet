use crate::ir::{IR, IRTrace, transform::IrTransform};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicaliseCommutativeInputs;

impl IrTransform for CanonicaliseCommutativeInputs {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        for op in ir.operations() {
            ir.get_op_mut(op.id())?.canonicalise_commutative_inputs()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        core::{Binary, DType},
        ir::graph::IrType,
    };

    #[test]
    fn commutative_inputs() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let ty = IrType::new(1, DType::F32);
        let x = ir.add_input(ty);
        let y = ir.add_input(ty);
        let z = ir.add_binary(x, y, Binary::Add)?;
        let w = ir.add_binary(y, x, Binary::Add)?;
        let t = ir.add_binary(w, z, Binary::Mul)?;

        ir.register_output(t);

        assert_eq!(ir.get_op(ir.get_parent_op(z)?)?.inputs(), &[x, y]);
        assert_eq!(ir.get_op(ir.get_parent_op(w)?)?.inputs(), &[y, x]);
        assert_eq!(ir.get_op(ir.get_parent_op(t)?)?.inputs(), &[w, z]);

        ir.transform(CanonicaliseCommutativeInputs)?;

        assert_eq!(ir.get_op(ir.get_parent_op(z)?)?.inputs(), &[x, y]);
        assert_eq!(ir.get_op(ir.get_parent_op(w)?)?.inputs(), &[x, y]);
        assert_eq!(ir.get_op(ir.get_parent_op(t)?)?.inputs(), &[z, w]);

        ir.check_valid()
    }
}
