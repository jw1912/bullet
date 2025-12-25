use crate::ir::{
    IrError, IrGraph,
    operation::{BroadcastAcrossDimension, IrBinary, IrOperation, IrUnary},
};

impl IrGraph {
    pub fn fold_broadcasts(&mut self) -> Result<(), IrError> {
        while self.fold_single_broadcast()? {
            self.eliminate_unused_ops()?;
        }
        Ok(())
    }

    fn fold_single_broadcast(&mut self) -> Result<bool, IrError> {
        for op in self.ops.values() {
            if let Some(unary) = IrOperation::downcast::<IrUnary>(op.op()) {
                let [output] = op.outputs()[..] else { panic!() };
                let [parent] = op.inputs()[..] else { panic!() };

                let parent_op = self.get_op(self.get_parent_op(parent)?)?;
                if let Some(broadcast) = IrOperation::downcast::<BroadcastAcrossDimension>(parent_op.op()).cloned() {
                    assert_eq!(parent_op.outputs()[..], [parent]);

                    let out_dtype = self.get_node_type(output)?.dtype();
                    let new_broadcast = broadcast.with_new_dtype(out_dtype);
                    let [grandparent] = parent_op.inputs()[..] else { panic!() };
                    let new_input = self.add_unary(grandparent, unary.op())?;
                    let new_output = self.add_op([new_input], new_broadcast)?;
                    self.swap_outputs(new_output[0], output)?;

                    return Ok(true);
                }
            }

            if let Some(binary) = IrOperation::downcast::<IrBinary>(op.op()) {
                let [output] = op.outputs()[..] else { panic!() };
                let [lparent, rparent] = op.inputs()[..] else { panic!() };

                let lparent_op = self.get_op(self.get_parent_op(lparent)?)?;
                let rparent_op = self.get_op(self.get_parent_op(rparent)?)?;
                if let (Some(lbroadcast), Some(rbroadcast)) = (
                    IrOperation::downcast::<BroadcastAcrossDimension>(lparent_op.op()).cloned(),
                    IrOperation::downcast::<BroadcastAcrossDimension>(rparent_op.op()).cloned(),
                ) {
                    assert_eq!(lparent_op.outputs()[..], [lparent]);
                    assert_eq!(rparent_op.outputs()[..], [rparent]);

                    let [lgrandparent] = lparent_op.inputs()[..] else { panic!() };
                    let [rgrandparent] = rparent_op.inputs()[..] else { panic!() };

                    let out_dtype = self.get_node_type(output)?.dtype();
                    let new_broadcast = lbroadcast.with_new_dtype(out_dtype);
                    if new_broadcast == rbroadcast.with_new_dtype(out_dtype) {
                        let new_input = self.add_binary(lgrandparent, rgrandparent, binary.op())?;
                        let new_output = self.add_op([new_input], new_broadcast)?;
                        self.swap_outputs(new_output[0], output)?;

                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        common::{DType, Size, Unary},
        ir::node::IrType,
    };

    use super::*;

    #[test]
    fn fold_broadcast_unary() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let a = ir.add_leaf(IrType::new(1, DType::F32));
        let broadcast = BroadcastAcrossDimension::new(DType::F32, [1], 0, Size::variable())?;
        let b1 = ir.add_op([a], broadcast)?[0];
        let c1 = ir.add_unary(b1, Unary::Sin)?;

        let b2 = ir.add_unary(a, Unary::Sin)?;
        let c2 = ir.add_op([b2], broadcast)?[0];

        ir.register_output(c1);
        ir.register_output(c2);

        ir.fold_broadcasts()?;
        ir.eliminate_common_subexprs()?;
        ir.eliminate_copies()?;

        assert_eq!(ir.num_ops(), 4);
        assert!(ir.is_copy(c1)? == Some(c2) || ir.is_copy(c2)? == Some(c1));

        Ok(())
    }

    #[test]
    fn fold_broadcast_complex() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let a = ir.add_leaf(IrType::new(1, DType::F32));
        let broadcast = BroadcastAcrossDimension::new(DType::F32, [1], 0, Size::variable())?;
        let b1 = ir.add_op([a], broadcast)?[0];
        let c1 = ir.add_unary(b1, Unary::Sin)?;

        let b2 = ir.add_unary(a, Unary::Sin)?;
        let c2 = ir.add_op([b2], broadcast)?[0];

        ir.register_output(c1);
        ir.register_output(c2);

        ir.fold_broadcasts()?;
        ir.eliminate_common_subexprs()?;
        ir.eliminate_copies()?;

        assert_eq!(ir.num_ops(), 4);
        assert!(ir.is_copy(c1)? == Some(c2) || ir.is_copy(c2)? == Some(c1));

        Ok(())
    }
}
