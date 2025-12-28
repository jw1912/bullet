use crate::ir::{
    IR, IRTrace,
    graph::operation::{BroadcastAcrossDimension, IrBinary, IrOperation, IrUnary},
    transform::{IrTransform, eliminate::EliminateUnusedOperations},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldBroadcasts;

impl IrTransform for FoldBroadcasts {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        while fold_single_broadcast(ir)? {
            ir.transform(EliminateUnusedOperations)?;
        }
        Ok(())
    }
}

fn fold_single_broadcast(ir: &mut IR) -> Result<bool, IRTrace> {
    for op in ir.operations() {
        if let Some(unary) = IrOperation::downcast::<IrUnary>(op.op()) {
            let [output] = op.outputs()[..] else { panic!() };
            let [parent] = op.inputs()[..] else { panic!() };

            let parent_op = ir.get_op(ir.get_parent_op(parent)?)?;
            if let Some(broadcast) = IrOperation::downcast::<BroadcastAcrossDimension>(parent_op.op()).cloned() {
                assert_eq!(parent_op.outputs()[..], [parent]);

                let out_dtype = ir.get_node(output)?.ty().dtype();
                let new_broadcast = broadcast.with_new_dtype(out_dtype);
                let [grandparent] = parent_op.inputs()[..] else { panic!() };
                let new_input = ir.add_unary(grandparent, unary.op())?;
                let new_output = ir.add_op([new_input], Ok::<_, IRTrace>(new_broadcast))?;
                ir.swap_outputs(new_output[0], output)?;

                return Ok(true);
            }
        }

        if let Some(binary) = IrOperation::downcast::<IrBinary>(op.op()) {
            let [output] = op.outputs()[..] else { panic!() };
            let [lparent, rparent] = op.inputs()[..] else { panic!() };

            let lparent_op = ir.get_op(ir.get_parent_op(lparent)?)?;
            let rparent_op = ir.get_op(ir.get_parent_op(rparent)?)?;
            if let (Some(lbroadcast), Some(rbroadcast)) = (
                IrOperation::downcast::<BroadcastAcrossDimension>(lparent_op.op()).cloned(),
                IrOperation::downcast::<BroadcastAcrossDimension>(rparent_op.op()).cloned(),
            ) {
                assert_eq!(lparent_op.outputs()[..], [lparent]);
                assert_eq!(rparent_op.outputs()[..], [rparent]);

                let [lgrandparent] = lparent_op.inputs()[..] else { panic!() };
                let [rgrandparent] = rparent_op.inputs()[..] else { panic!() };

                let out_dtype = ir.get_node(output)?.ty().dtype();
                let new_broadcast = lbroadcast.with_new_dtype(out_dtype);
                if new_broadcast == rbroadcast.with_new_dtype(out_dtype) {
                    let new_input = ir.add_binary(lgrandparent, rgrandparent, binary.op())?;
                    let new_output = ir.add_op([new_input], Ok::<_, IRTrace>(new_broadcast))?;
                    ir.swap_outputs(new_output[0], output)?;

                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use crate::{
        core::{Binary, DType, Size, Unary},
        ir::graph::IrType,
    };

    use super::*;

    #[test]
    fn fold_broadcast_unary() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let a = ir.add_input(IrType::new(1, DType::F32));
        let broadcast = BroadcastAcrossDimension::new(DType::F32, [1], 0, Size::variable());
        let b = ir.add_op([a], broadcast.clone())?[0];
        let c = ir.add_unary(b, Unary::Sin)?;
        let d = ir.add_unary(c, Unary::Exp)?;

        ir.register_output(d);
        ir.transform(FoldBroadcasts)?;

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
        ir.transform(FoldBroadcasts)?;

        assert_eq!(ir.parent_op(e)?, Some(&broadcast?));

        ir.check_valid()
    }
}
