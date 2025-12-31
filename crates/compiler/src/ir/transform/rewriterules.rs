use std::fmt;

use crate::{
    core::{CABinary, DTypeValue},
    ir::{
        IR, IRTrace,
        graph::IrOperation,
        operation::{BroadcastAcrossDimension, CABinaryOp, ScalarConstant, UnaryOp},
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
    rewrites op (unary = [UnaryOp] (broadcast = [BroadcastAcrossDimension] (grandparent)))
    {
        let new_broadcast = broadcast.with_new_dtype(unary.output_type().dtype());
        let new_parent = ir.add_unary(grandparent.id(), unary.op())?;
        ir.replace_operation(op.id(), [new_parent], new_broadcast)?;
        return Ok(true);
    }
}

rewriterule! {
    rulename BroadcastBinaryIntoBinaryBroadcast on ir
    rewrites op (binary = [CABinaryOp]
        (lb = [BroadcastAcrossDimension] (lgp))
        (rb = [BroadcastAcrossDimension] (rgp)))
    {
        let out_dtype = binary.ty().dtype();
        let new_broadcast = lb.with_new_dtype(out_dtype);
        if new_broadcast == rb.with_new_dtype(out_dtype) {
            let new_parent = ir.add_binary(lgp.id(), rgp.id(), binary.op())?;
            ir.replace_operation(op.id(), [new_parent], new_broadcast)?;
            return Ok(true);
        }
    }
}

rewriterule! {
    rulename ScalarBroadcastBinaryIntoBinaryBroadcast on ir
    rewrites op (bi = [CABinaryOp] (sc = [ScalarConstant]) (br = [BroadcastAcrossDimension] (gp)))
    {
        let out_dtype = bi.ty().dtype();
        let new_broadcast = br.with_new_dtype(out_dtype);
        let rhs = gp.id();
        let new_scalar = ir.add_scalar(sc.0, gp.ty().size());
        let new_parent = ir.add_binary(new_scalar, rhs, bi.op())?;
        ir.replace_operation(op.id(), [new_parent], new_broadcast)?;
        return Ok(true);
    }
}

rewriterule! {
    rulename ExpandDistributive on ir
    rewrites op [
        (z = [CABinaryOp] (x = [CABinaryOp] (a) (b)) (y))
        (z = [CABinaryOp] (y) (x = [CABinaryOp] (a) (b)))
    ] {
        if z.op() == CABinary::Mul && x.op() == CABinary::Add {
            let new_op = CABinaryOp::new(a.ty(), CABinary::Add)?;
            let (y, a, b) = (y.id(), a.id(), b.id());
            let new_lhs = ir.add_binary(a, y, CABinary::Mul)?;
            let new_rhs = ir.add_binary(b, y, CABinary::Mul)?;
            ir.replace_operation(op.id(), [new_lhs, new_rhs], new_op)?;
            return Ok(true);
        }
    }
}

rewriterule! {
    rulename FactoriseDistributive on ir
    rewrites op (z = [CABinaryOp] (x = [CABinaryOp] (a) (b)) (y = [CABinaryOp] (c) (d)))
    {
        if z.op() == CABinary::Add && x.op() == CABinary::Mul && y.op() == CABinary::Mul {
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
                        let mul = CABinaryOp::new(ty, CABinary::Mul)?;
                        let add = ir.add_binary(x, y, CABinary::Add)?;
                        ir.replace_operation(op_id, [lhs, add], mul)?;
                        return Ok(true);
                    }
                }
            }
        }
    }
}

rewriterule! {
    rulename CombineXAddMulX on ir
    rewrites op [
        (bin1 = [CABinaryOp] (bin2 = [CABinaryOp] (scalar = [ScalarConstant]) (x)) (y))
        (bin1 = [CABinaryOp] (y) (bin2 = [CABinaryOp] (scalar = [ScalarConstant]) (x)))
    ] {
        println!("found pattern!");
        if bin1.op() == CABinary::Add && bin2.op() == CABinary::Mul && x.id() == y.id() {
            let new_value = CABinary::Add.evaluate(scalar.0, DTypeValue::one(scalar.0.dtype())).unwrap();
            let new_op = CABinaryOp::new(x.ty(), CABinary::Mul)?;
            let x = x.id();
            let new_scalar = ir.add_scalar(new_value, scalar.1);
            ir.replace_operation(op.id(), [new_scalar, x], new_op)?;
            return Ok(true);
        }
    }
}

rewriterule! {
    rulename CombineMulXAddMulX on ir
    rewrites op (bin1 = [CABinaryOp]
        (bin2 = [CABinaryOp] (scalar1 = [ScalarConstant]) (x))
        (bin3 = [CABinaryOp] (scalar2 = [ScalarConstant]) (y)))
    {
        if bin1.op() == CABinary::Add
            && bin2.op() == CABinary::Mul
            && bin3.op() == CABinary::Mul
            && x.id() == y.id()
        {
            let new_value = CABinary::Add.evaluate(scalar1.0, scalar2.0).unwrap();
            let new_op = CABinaryOp::new(x.ty(), CABinary::Mul)?;
            let x = x.id();
            let new_scalar = ir.add_scalar(new_value, scalar1.1);
            ir.replace_operation(op.id(), [new_scalar, x], new_op)?;
            return Ok(true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        core::{CABinary, DType, Size, Unary},
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

        let d = ir.add_binary(c1, c2, CABinary::Add)?;
        let e = ir.add_binary(d, d, CABinary::Mul)?;

        ir.register_output(e);
        let nest = RewriteNest(BroadcastUnaryIntoUnaryBroadcast, BroadcastBinaryIntoBinaryBroadcast);
        ir.transform(RewritePass(nest))?;

        assert_eq!(ir.parent_op(e)?, Some(&broadcast?));

        ir.check_valid()
    }

    #[test]
    fn fold_scalar_broadcast_binary() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let size = Size::variable();
        let broadcast = BroadcastAcrossDimension::new(DType::F32, [1], 0, size);

        let a = ir.add_input(IrType::new(1, DType::F32));
        let b = ir.add_scalar(1.0, size);
        let c = ir.add_op([a], broadcast.clone())?[0];

        let d = ir.add_binary(b, c, CABinary::Add)?;

        ir.register_output(d);
        let nest = RewriteNest(BroadcastBinaryIntoBinaryBroadcast, ScalarBroadcastBinaryIntoBinaryBroadcast);
        ir.transform(RewritePass(nest))?;
        assert_eq!(ir.parent_op(d)?, Some(&broadcast?));

        ir.check_valid()
    }
}
