use std::rc::Rc;

use crate::{
    core::DTypeTensor,
    ir::{
        IR, IRTrace,
        graph::{IrNodeId, IrOperationId, IrOperationType},
        operation::{Constant, ScalarConstant},
        transform::{IrTransform, eliminate::*, foldrules::*, ordering::*, rewriterules::*},
    },
};

#[derive(Clone, Debug, Default)]
pub struct CanonicalisePass {
    cleanups: Vec<Rc<dyn IrTransform>>,
    folds: Vec<Rc<dyn FoldRule>>,
    rewrites: Vec<Rc<dyn RewriteRule>>,
}

impl CanonicalisePass {
    /// Canonicalisation pass that contains only rewrites that
    /// are always objectively good, so other canonicalisation
    /// passes can build on top of it.
    pub fn base() -> Self {
        Self::default()
            .add_cleanup(OrderCommutativeInputs)
            .add_cleanup(EliminateCopies)
            .add_cleanup(EliminateUnusedOperations)
            .add_cleanup(EliminateCommonSubExpressions)
            .add_fold(EvalScalarConstUnary)
            .add_fold(EvalScalarConstBinary)
            .add_fold(FoldFixedSizeScalarConst)
            .add_fold(FoldVarSizeScalarConst)
            .add_fold(FoldConstIdentities)
            .add_fold(FoldMulByZero)
            .add_rewrite(BroadcastUnaryIntoUnaryBroadcast)
            .add_rewrite(BroadcastBinaryIntoBinaryBroadcast)
            .add_rewrite(ScalarBroadcastBinaryIntoBinaryBroadcast)
            .add_rewrite(CombineXAddMulX)
            .add_rewrite(CombineMulXAddMulX)
    }

    /// Canonicalisation pass that expands all distributive rules.
    /// This ensures that all common subexpressions are eliminated.
    ///
    /// Otherwise expressions like `x*x - y*y` and `(x - y)(x + y)`
    /// would be difficult to identify as equivalent (without hardcoding
    /// such a rule)
    pub fn expand() -> Self {
        Self::base().add_rewrite(ExpandDistributive)
    }

    /// Canonicalisation pass that factorises all distributive rules.
    pub fn factorise() -> Self {
        Self::base().add_rewrite(FactoriseDistributive)
    }

    pub fn add_cleanup(mut self, cleanup: impl IrTransform) -> Self {
        self.cleanups.push(Rc::new(cleanup));
        self
    }

    pub fn add_fold(mut self, fold: impl FoldRule) -> Self {
        self.folds.push(Rc::new(fold));
        self
    }

    pub fn add_rewrite(mut self, rewrite: impl RewriteRule) -> Self {
        self.rewrites.push(Rc::new(rewrite));
        self
    }

    pub fn apply_single_fold(&self, ir: &mut IR, mut id: IrOperationId) -> Result<bool, IRTrace> {
        let mut success = false;

        'fold: loop {
            let op = ir.get_op(id)?;

            if let Some(consts) = Self::try_evaluate(ir, op.inputs(), op.op())? {
                let outputs = op.outputs().to_vec();
                for (old, new_const) in outputs.into_iter().zip(consts) {
                    let new = ir.add_op([], Ok::<_, IRTrace>(new_const))?[0];
                    ir.swap_outputs(old, new)?;
                }

                ir.remove_op(id)?;
                return Ok(true);
            }

            for fold in &self.folds {
                if let Some(new) = fold.fold(ir, op)? {
                    id = ir.replace_op(op.id(), new)?;
                    success = true;
                    continue 'fold;
                }
            }

            let op = op.clone();
            for rewrite in &self.rewrites {
                if rewrite.apply(ir, op.clone())? {
                    return Ok(true);
                }
            }

            return Ok(success);
        }
    }

    fn try_evaluate(
        ir: &IR,
        inputs: &[IrNodeId],
        op: &Rc<dyn IrOperationType>,
    ) -> Result<Option<Vec<Constant>>, IRTrace> {
        if !inputs.is_empty() {
            let mut consts = Vec::new();

            for &input in inputs {
                let parent = ir.get_op(ir.get_parent_op(input)?)?;

                if let Some(Constant(value)) = parent.downcast().cloned() {
                    consts.push(value);
                } else if let Some(Some(scalar)) = parent.downcast().map(ScalarConstant::to_tensor) {
                    consts.push(scalar);
                } else {
                    return Ok(None);
                }
            }

            let mut tensors = Vec::new();
            for &ty in &op.outputs() {
                if let Some(size) = ty.size().evaluate_constant() {
                    tensors.push(DTypeTensor::new(ty.dtype(), size));
                } else {
                    return Ok(None);
                }
            }

            let inputs = consts.iter().collect::<Vec<_>>();
            let mut outputs = tensors.iter_mut().collect::<Vec<_>>();

            op.evaluate(&inputs, &mut outputs);

            return Ok(Some(tensors.into_iter().map(Constant).collect()));
        }

        Ok(None)
    }
}

impl IrTransform for CanonicalisePass {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        let mut success = true;

        while success {
            for cleanup in &self.cleanups {
                ir.transform_dyn(cleanup.clone())?;
            }

            success = false;

            for op in ir.ordered_operations()? {
                success |= self.apply_single_fold(ir, op.id())?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        core::{CABinary, DType, Size, Unary},
        ir::{
            graph::IrType,
            operation::{BroadcastAcrossDimension, ScalarConstant},
        },
    };

    #[test]
    fn test_complex() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let size = Size::variable();

        let a = ir.add_scalar(1, 1);
        let ba = BroadcastAcrossDimension::new(DType::I32, [1], 0, size);
        let a = ir.add_op([a], ba)?[0];

        let b = ir.add_input(IrType::new(1, DType::F32));
        let bb = BroadcastAcrossDimension::new(DType::F32, [1], 0, size);
        let b = ir.add_op([b], bb.clone())?[0];

        let c = ir.add_unary(a, Unary::Cast(DType::F32))?;
        let d = ir.add_binary(c, b, CABinary::Add)?;

        let minus_one = ir.add_scalar(-1.0, size);
        let neg_d = ir.add_binary(minus_one, d, CABinary::Mul)?;
        let e = ir.add_binary(neg_d, d, CABinary::Add)?;

        ir.register_output(d);
        ir.register_output(e);

        ir.transform(CanonicalisePass::base())?;

        assert_eq!(ir.parent_op(d)?, Some(&bb?), "{ir}");
        assert_eq!(ir.parent_op(e)?, Some(&ScalarConstant(0.0.into(), size)), "{ir}");

        ir.check_valid()
    }
}
