use std::{fmt, rc::Rc};

use crate::ir::{
    IR, IRTrace,
    graph::IrOperation,
    operation::{BroadcastAcrossDimension, IrBinary, IrUnary},
    transform::{IrTransform, modify::AddOperation},
};

pub trait DestructiveRule: fmt::Debug + 'static {
    fn apply(&self, ir: &mut IR, operation: IrOperation) -> Result<bool, IRTrace>;
}

#[derive(Debug)]
pub struct DestructiveNest<A, B>(A, B);

impl<A: DestructiveRule, B: DestructiveRule> DestructiveRule for DestructiveNest<A, B> {
    fn apply(&self, ir: &mut IR, operation: IrOperation) -> Result<bool, IRTrace> {
        Ok(self.0.apply(ir, operation.clone())? || self.1.apply(ir, operation)?)
    }
}

#[derive(Debug)]
pub struct DestructivePass<T>(T);

impl<T: DestructiveRule> IrTransform for DestructivePass<T> {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        'outer: loop {
            for op in ir.operations() {
                if self.0.apply(ir, op.clone())? {
                    continue 'outer;
                }
            }

            return Ok(());
        }
    }
}

macro_rules! destructiverule {
    {
        rulename $name:ident on $irname:ident
        rewrites $op:ident ($($pattern:tt)*)
        $($tail:tt)*
    } => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;

        impl DestructiveRule for $name {
            fn apply(
                &self,
                $irname: &mut IR,
                $op: IrOperation,
            ) -> Result<bool, IRTrace> {
                $crate::if_find_and_bind_pattern!(
                    $irname,
                    &$op,
                    ($($pattern)*),
                    $($tail)*
                );

                Ok(false)
            }
        }
    };
}

destructiverule! {
    rulename BroadcastUnaryIntoUnaryBroadcast on ir
    rewrites op (unary = [IrUnary] (broadcast = [BroadcastAcrossDimension] (grandparent)))
    {
        let new_broadcast = broadcast.with_new_dtype(unary.output_type().dtype());
        let new_parent = ir.add_unary(grandparent.id(), unary.op())?;
        let new_op = AddOperation(vec![new_parent], Ok::<_, IRTrace>(Rc::new(new_broadcast)));
        ir.replace_op(op.id(), new_op)?;
        return Ok(true);
    }
}

destructiverule! {
    rulename BroadcastBinaryIntoBinaryBroadcast on ir
    rewrites op (binary = [IrBinary]
        (lb = [BroadcastAcrossDimension] (lgp))
        (rb = [BroadcastAcrossDimension] (rgp)))
    {
        let out_dtype = binary.output_type().dtype();
        let new_broadcast = lb.with_new_dtype(out_dtype);
        if new_broadcast == rb.with_new_dtype(out_dtype) {
            let new_parent = ir.add_binary(lgp.id(), rgp.id(), binary.op())?;
            let new_op = Rc::new(new_broadcast);
            let new_op = AddOperation(vec![new_parent], Ok::<_, IRTrace>(new_op));
            ir.replace_op(op.id(), new_op)?;
            return Ok(true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        core::{Binary, DType, Size, Unary},
        ir::graph::IrType,
    };

    #[test]
    fn fold_broadcast_unary() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let a = ir.add_input(IrType::new(1, DType::F32));
        let broadcast = BroadcastAcrossDimension::new(DType::F32, [1], 0, Size::variable());
        let b = ir.add_op([a], broadcast.clone())?[0];
        let c = ir.add_unary(b, Unary::Sin)?;
        let d = ir.add_unary(c, Unary::Exp)?;

        ir.register_output(d);
        ir.transform(DestructivePass(BroadcastUnaryIntoUnaryBroadcast))?;

        assert_eq!(ir.parent_op(d)?, Some(&broadcast?));

        ir.check_valid()
    }

    #[test]
    fn fold_broadcast_binary() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let a1 = ir.add_input(IrType::new(1, DType::F32));
        let broadcast = BroadcastAcrossDimension::new(DType::F32, [1], 0, Size::variable());
        let b1 = ir.add_op([a1], broadcast.clone())?[0];
        let c1 = ir.add_unary(b1, Unary::Sin)?;

        let a2 = ir.add_input(IrType::new(1, DType::I32));
        let b2 = ir.add_op([a2], broadcast.clone().map(|x| x.with_new_dtype(DType::I32)))?[0];
        let c2 = ir.add_unary(b2, Unary::Cast(DType::F32))?;

        let d = ir.add_binary(c1, c2, Binary::Add)?;
        let e = ir.add_binary(d, d, Binary::Mul)?;

        ir.register_output(e);
        let nest = DestructiveNest(BroadcastUnaryIntoUnaryBroadcast, BroadcastBinaryIntoBinaryBroadcast);
        ir.transform(DestructivePass(nest))?;

        assert_eq!(ir.parent_op(e)?, Some(&broadcast?));

        ir.check_valid()
    }
}
