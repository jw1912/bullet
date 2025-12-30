use std::fmt;

use crate::{
    core::Binary,
    ir::{
        IR, IRTrace,
        graph::IrOperation,
        operation::{BroadcastAcrossDimension, IrBinary, IrUnary},
        transform::IrTransform,
    },
};

pub trait RewriteRule: fmt::Debug + 'static {
    fn apply(&self, ir: &mut IR, operation: IrOperation) -> Result<bool, IRTrace>;
}

#[derive(Debug)]
pub struct RewriteNest<A, B>(A, B);

impl<A: RewriteRule, B: RewriteRule> RewriteRule for RewriteNest<A, B> {
    fn apply(&self, ir: &mut IR, operation: IrOperation) -> Result<bool, IRTrace> {
        Ok(self.0.apply(ir, operation.clone())? || self.1.apply(ir, operation)?)
    }
}

#[derive(Debug)]
pub struct RewritePass<T>(T);

impl<T: RewriteRule> IrTransform for RewritePass<T> {
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

macro_rules! rewriterule {
    {
        rulename $name:ident on $irname:ident
        rewrites $op:ident ($($pattern:tt)*)
        $tail:stmt
    } => {
        rewriterule! {
            rulename $name on $irname
            rewrites $op [($($pattern)*)]
            $tail
        }
    };
    {
        rulename $name:ident on $irname:ident
        rewrites $op:ident [
            $(($($pattern:tt)*))+
        ]
        $tail:stmt
    } => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;

        impl RewriteRule for $name {
            fn apply(
                &self,
                $irname: &mut IR,
                $op: IrOperation,
            ) -> Result<bool, IRTrace> {
                $($crate::if_find_and_bind_pattern!(
                    $irname,
                    &$op,
                    ($($pattern)*),
                    $tail
                );)+

                Ok(false)
            }
        }
    };
}

rewriterule! {
    rulename BroadcastUnaryIntoUnaryBroadcast on ir
    rewrites op (unary = [IrUnary] (broadcast = [BroadcastAcrossDimension] (grandparent)))
    {
        let new_broadcast = broadcast.with_new_dtype(unary.output_type().dtype());
        let new_parent = ir.add_unary(grandparent.id(), unary.op())?;
        ir.replace_operation(op.id(), [new_parent], new_broadcast)?;
        return Ok(true);
    }
}

rewriterule! {
    rulename BroadcastBinaryIntoBinaryBroadcast on ir
    rewrites op (binary = [IrBinary]
        (lb = [BroadcastAcrossDimension] (lgp))
        (rb = [BroadcastAcrossDimension] (rgp)))
    {
        let out_dtype = binary.output_type().dtype();
        let new_broadcast = lb.with_new_dtype(out_dtype);
        if new_broadcast == rb.with_new_dtype(out_dtype) {
            let new_parent = ir.add_binary(lgp.id(), rgp.id(), binary.op())?;
            ir.replace_operation(op.id(), [new_parent], new_broadcast)?;
            return Ok(true);
        }
    }
}

rewriterule! {
    rulename ExpandDistributive on ir
    rewrites op [
        (z = [IrBinary] (x = [IrBinary] (a) (b)) (y))
        (z = [IrBinary] (y) (x = [IrBinary] (a) (b)))
    ] {
        if z.op() == Binary::Mul && x.op() == Binary::Add {
            let new_op = IrBinary::new(a.ty(), b.ty(), Binary::Add)?;
            let (y, a, b) = (y.id(), a.id(), b.id());
            let new_lhs = ir.add_binary(a, y, Binary::Mul)?;
            let new_rhs = ir.add_binary(b, y, Binary::Mul)?;
            ir.replace_operation(op.id(), [new_lhs, new_rhs], new_op)?;
            return Ok(true);
        }
    }
}

rewriterule! {
    rulename FactoriseDistributive on ir
    rewrites op (z = [IrBinary] (x = [IrBinary] (a) (b)) (y = [IrBinary] (c) (d)))
    {
        if z.op() == Binary::Add && x.op() == Binary::Mul && y.op() == Binary::Mul {
            let [xn, yn] = op.inputs()[..] else { unreachable!() };
            if ir.get_node(xn)?.children() == 1 && ir.get_node(yn)?.children() == 1 {
                let ty = a.ty();
                let (a, b, c, d) = (a.id(), b.id(), c.id(), d.id());
                for ((lhs, mat), (x, y)) in [
                    ((a, c), (b, d)),
                    ((a, d), (b, c)),
                    ((b, c), (a, d)),
                    ((b, d), (a, c)),
                ] {
                    if lhs == mat {
                        let op_id = op.id();
                        let mul = IrBinary::new(ty, ty, Binary::Mul)?;
                        let add = ir.add_binary(x, y, Binary::Add)?;
                        ir.replace_operation(op_id, [lhs, add], mul)?;
                        return Ok(true);
                    }
                }
            }
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
        ir.transform(RewritePass(BroadcastUnaryIntoUnaryBroadcast))?;

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
        let nest = RewriteNest(BroadcastUnaryIntoUnaryBroadcast, BroadcastBinaryIntoBinaryBroadcast);
        ir.transform(RewritePass(nest))?;

        assert_eq!(ir.parent_op(e)?, Some(&broadcast?));

        ir.check_valid()
    }
}
