mod constant;
mod rules;

use std::{fmt, rc::Rc};

pub(crate) use rules::foldrule;

use crate::{
    core::{Binary, DTypeValue, Unary},
    ir::{
        IR, IRTrace,
        graph::{
            IrNodeId, IrOperation, IrOperationId, IrOperationType, IrType,
            operation::{BroadcastAcrossDimension, Constant, IrBinary, IrCopy, IrUnary, ScalarConstant},
        },
        transform::{IrTransform, modify::AddOperation},
    },
};

pub trait Fold: fmt::Debug + 'static {
    fn fold(
        &self,
        ir: &IR,
        inputs: &[IrNodeId],
        operation: &Rc<dyn IrOperationType>,
    ) -> Result<Option<AddOperation>, IRTrace>;
}

#[derive(Clone, Debug, Default)]
pub struct FoldPass(Vec<Rc<dyn Fold>>);

impl<F: Fold> From<F> for FoldPass {
    fn from(value: F) -> Self {
        Self(vec![Rc::new(value)])
    }
}

impl FoldPass {
    pub fn all() -> Self {
        Self::default()
            .add_fold(FoldScalarConstIntoUnary)
            .add_fold(FoldScalarConstLhsIntoBinary)
            .add_fold(FoldScalarConstRhsIntoBinary)
            .add_fold(FoldNegLhsIntoAdd)
            .add_fold(FoldNegRhsIntoAdd)
            .add_fold(FoldFixedSizeScalarConst)
            .add_fold(FoldVarSizeScalarConst)
            .add_fold(FoldXMinusXIntoZero)
            .add_fold(FoldMulByOne)
            .add_fold(FoldDivByOne)
            .add_fold(FoldAddZero)
            .add_fold(FoldSubZero)
    }

    pub fn add_fold(mut self, fold: impl Fold) -> Self {
        self.0.push(Rc::new(fold));
        self
    }

    pub fn apply_single_fold(&self, ir: &mut IR, mut id: IrOperationId) -> Result<bool, IRTrace> {
        let mut success = false;

        'fold: loop {
            let op = ir.get_op(id)?;

            if let Some(consts) = constant::fold_constant_expression(ir, op.inputs(), op.op())? {
                let outputs = op.outputs().to_vec();
                for (old, new_const) in outputs.into_iter().zip(consts) {
                    let new = ir.add_op([], Ok::<_, IRTrace>(new_const))?[0];
                    ir.swap_outputs(old, new)?;
                }

                ir.remove_op(id)?;
                return Ok(true);
            }

            for fold in &self.0 {
                if let Some(new) = fold.fold(ir, op.inputs(), op.op())? {
                    id = ir.replace_op(op.id(), new)?;
                    success = true;
                    continue 'fold;
                }
            }

            return Ok(success);
        }
    }
}

impl IrTransform for FoldPass {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        let mut success = true;

        while success {
            ir.eliminate_dead_ops()?;

            success = false;

            for op in ir.ordered_operations()? {
                success |= self.apply_single_fold(ir, op.id())?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
use crate::core::{DType, DTypeTensor, Size};

foldrule! {
    rulename FoldFixedSizeScalarConst on _ir
    rewrites () -> Constant(value),
    into () -> ScalarConstant(scalar, value.size().into()),
    iff {
        Some(scalar) = value.scalar()
    }
    testcase fixed_size_scalar_const |ir| {
        ir.add_const(DTypeTensor::I32(vec![1; 16]))
    }
}

foldrule! {
    rulename FoldVarSizeScalarConst on ir
    rewrites (input) -> broadcast = BroadcastAcrossDimension,
    into () -> ScalarConstant(*val, broadcast.outputs()[0].size()),
    iff {
        Some(ScalarConstant(val, _)) = ir.parent_op(input.id())?
    }
    testcase var_size_scalar_const |ir| {
        let a = ir.add_scalar(1, 1);
        ir.add_broadcast(a, [1], 0, Size::variable())?
    }
}

foldrule! {
    rulename FoldScalarConstLhsIntoBinary on ir
    rewrites (a, b) -> binary = IrBinary,
    into (b) -> {
        let ty = IrType::new(size, val.dtype());
        let op = Unary::BinaryWithConst { op: binary.op(), val, lhs: false };
        IrUnary::new(ty, op)?
    },
    iff {
        Some(ScalarConstant(val, size)) = ir.parent_op(a.id())?.cloned()
    }
    testcase scalar_const_lhs_into_binary |ir| {
        let size = Size::variable();
        let a = ir.add_input(IrType::new(size, DType::F32));
        let b = ir.add_scalar(1.0, size);
        ir.add_binary(b, a, Binary::Add)?
    }
}

foldrule! {
    rulename FoldScalarConstRhsIntoBinary on ir
    rewrites (a, b) -> binary = IrBinary,
    into (a) -> {
        let ty = IrType::new(size, val.dtype());
        let op = Unary::BinaryWithConst { op: binary.op(), val, lhs: true };
        IrUnary::new(ty, op).unwrap()
    },
    iff {
        Some(ScalarConstant(val, size)) = ir.parent_op(b.id())?.cloned()
    }
    testcase scalar_const_rhs_into_binary |ir| {
        let size = Size::variable();
        let a = ir.add_input(IrType::new(size, DType::F32));
        let b = ir.add_scalar(1.0, size);
        ir.add_binary(a, b, Binary::Add)?
    }
}

foldrule! {
    rulename FoldScalarConstIntoUnary on ir
    rewrites (a) -> unary = IrUnary,
    into () -> ScalarConstant(unary.op().evaluate(val).unwrap(), size),
    iff {
        Some(ScalarConstant(val, size)) = ir.parent_op(a.id())?.cloned()
    }
}

foldrule! {
    rulename FoldNegLhsIntoAdd on ir
    rewrites (a, b) -> binary = IrBinary,
    into (b, a) -> IrBinary::new(b.ty(), a.ty(), Binary::Sub)?,
    iff {{
        if binary.op() == Binary::Add
            && let Some(unary) = ir.parent_op::<IrUnary>(a.id())?
            && let Unary::BinaryWithConst { val, op: Binary::Mul, .. } = unary.op()
            && (val == DTypeValue::F32(-1.0) || val == DTypeValue::I32(-1))
        {
            true
        } else {
            false
        }
    }}
}

foldrule! {
    rulename FoldNegRhsIntoAdd on ir
    rewrites (a, b) -> binary = IrBinary,
    into (a, b) -> IrBinary::new(a.ty(), b.ty(), Binary::Sub)?,
    iff {{
        if binary.op() == Binary::Add
            && let Some(unary) = ir.parent_op::<IrUnary>(b.id())?
            && let Unary::BinaryWithConst { val, op: Binary::Mul, .. } = unary.op()
            && (val == DTypeValue::F32(-1.0) || val == DTypeValue::I32(-1))
        {
            true
        } else {
            false
        }
    }}
}

foldrule! {
    rulename FoldXMinusXIntoZero on ir
    rewrites (a, b) -> binary = IrBinary,
    into () -> {
        let ty = a.ty();
        ScalarConstant(DTypeValue::zero(ty.dtype()), ty.size())
    },
    iff {
        binary.op() == Binary::Sub && a.id() == b.id()
    }
}

macro_rules! elision_foldrule {
    ($name:ident, $op:ident, $float:expr, $int:expr) => {
        foldrule! {
            rulename $name on ir
            rewrites (a) -> unary = IrUnary,
            into (a) -> IrCopy(a.ty()),
            iff {{
                if let Unary::BinaryWithConst { op: Binary::$op, val, lhs: true } = unary.op() {
                    val == $float.into() || val == $int.into()
                } else {
                    false
                }
            }}
        }
    };
}

elision_foldrule!(FoldMulByOne, Mul, 1.0, 1);
elision_foldrule!(FoldDivByOne, Div, 1.0, 1);
elision_foldrule!(FoldAddZero, Add, 0.0, 0);
elision_foldrule!(FoldSubZero, Sub, 0.0, 0);

#[cfg(test)]
mod tests {
    use crate::core::{DType, DTypeValue, Size};

    use super::*;

    #[test]
    fn test_complex() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let size = Size::variable();

        let a = ir.add_const(DTypeValue::I32(1).into());
        let ba = BroadcastAcrossDimension::new(DType::I32, [1], 0, size);
        let a = ir.add_op([a], ba)?[0];

        let b = ir.add_input(IrType::new(1, DType::F32));
        let bb = BroadcastAcrossDimension::new(DType::F32, [1], 0, size);
        let b = ir.add_op([b], bb)?[0];

        let c = ir.add_unary(a, Unary::Cast(DType::F32))?;
        let d = ir.add_binary(c, b, Binary::Add)?;
        let e = ir.add_binary(d, d, Binary::Sub)?;

        ir.register_output(d);
        ir.register_output(e);
        ir.transform(FoldPass::all())?;
        ir.eliminate_dead_ops()?;

        let ty = IrType::new(size, DType::F32);
        let op = Unary::BinaryWithConst { op: Binary::Add, val: 1.0.into(), lhs: true };
        assert_eq!(ir.parent_op(d)?, Some(&IrUnary::new(ty, op)?));
        assert_eq!(ir.parent_op(e)?, Some(&ScalarConstant(0.0.into(), size)));

        ir.check_valid()
    }
}
