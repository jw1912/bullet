mod constant;
mod rules;

use std::{fmt, rc::Rc};

use crate::{
    core::{Binary, DType, DTypeValue, Unary},
    foldrule,
    ir::{
        IR, IRTrace,
        graph::{
            IrOperation, IrOperationId, IrType,
            operation::{BroadcastAcrossDimension, Constant, IrBinary, IrCopy, IrUnary, ScalarConstant},
        },
        transform::{IrTransform, modify::AddOperation},
    },
};

pub trait Fold: fmt::Debug + 'static {
    fn fold(&self, ir: &IR, operation: &IrOperation) -> Result<Option<AddOperation>, IRTrace>;
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
            .add_fold(FoldConstIdentities)
            .add_fold(FoldSubFromZero)
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
                if let Some(new) = fold.fold(ir, op)? {
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
use crate::core::{DTypeTensor, Size};

foldrule! {
    rulename FoldFixedSizeScalarConst on ir
    rewrites (constant = [Constant])
    into [ScalarConstant(scalar, constant.0.size().into())]
    given {
        Some(scalar) = constant.0.scalar();
    }
    testcase fixed_size_scalar_const {
        ir.add_const(DTypeTensor::I32(vec![1; 16]))
    }
}

foldrule! {
    rulename FoldVarSizeScalarConst on ir
    rewrites (broadcast = [BroadcastAcrossDimension] (input = [ScalarConstant]))
    into [ScalarConstant(input.0, broadcast.output_size())]
    testcase var_size_scalar_const {
        let a = ir.add_scalar(1, 1);
        ir.add_broadcast(a, [1], 0, Size::variable())?
    }
}

foldrule! {
    rulename FoldScalarConstLhsIntoBinary on ir
    rewrites (binary = [IrBinary] (a = [ScalarConstant]) (b))
    into [{
        let ScalarConstant(val, size) = *a;
        let ty = IrType::new(size, val.dtype());
        let op = Unary::BinaryWithConst { op: binary.op(), val, lhs: false };
        IrUnary::new(ty, op)?
    }] (b)
    testcase scalar_const_lhs_into_binary {
        let size = Size::variable();
        let a = ir.add_input(IrType::new(size, DType::F32));
        let b = ir.add_scalar(1.0, size);
        ir.add_binary(b, a, Binary::Add)?
    }
}

foldrule! {
    rulename FoldScalarConstRhsIntoBinary on ir
    rewrites (binary = [IrBinary] (a) (b = [ScalarConstant]))
    into [{
        let ScalarConstant(val, size) = *b;
        let ty = IrType::new(size, val.dtype());
        let op = Unary::BinaryWithConst { op: binary.op(), val, lhs: true };
        IrUnary::new(ty, op).unwrap()
    }] (a)
    testcase scalar_const_rhs_into_binary {
        let size = Size::variable();
        let a = ir.add_input(IrType::new(size, DType::F32));
        let b = ir.add_scalar(1.0, size);
        ir.add_binary(a, b, Binary::Add)?
    }
}

foldrule! {
    rulename FoldScalarConstIntoUnary on ir
    rewrites (unary = [IrUnary] (scalar = [ScalarConstant]))
    into [ScalarConstant(unary.op().evaluate(scalar.0).unwrap(), scalar.1)]
}

foldrule! {
    rulename FoldNegLhsIntoAdd on _ir
    rewrites (binary = [IrBinary] (unary = [IrUnary] (a)) (b))
    into [IrBinary::new(b.ty(), a.ty(), Binary::Sub)?] (b) (a)
    given {{
        if binary.op() == Binary::Add
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
    rewrites (binary = [IrBinary] (a) (unary = [IrUnary] (b)))
    into [IrBinary::new(a.ty(), b.ty(), Binary::Sub)?] (a) (b)
    given {{
        if binary.op() == Binary::Add
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
    rewrites (binary = [IrBinary] (a) (b))
    into [{
        let ty = a.ty();
        ScalarConstant(DTypeValue::zero(ty.dtype()), ty.size())
    }]
    given { binary.op() == Binary::Sub && a.id() == b.id() }
}

foldrule! {
    rulename FoldConstIdentities on ir
    rewrites (unary = [IrUnary] (a))
    into [IrCopy(a.ty())] (a)
    given {{
        if let Unary::BinaryWithConst { op, val, lhs } = unary.op() {
            match op {
                Binary::Mul => val == 1.0.into() || val == 1.into(),
                Binary::Add => val == 0.0.into() || val == 0.into(),
                Binary::Div => lhs && (val == 1.0.into() || val == 1.into()),
                Binary::Sub => lhs && (val == 0.0.into() || val == 0.into()),
                _ => false,
            }
        } else {
            false
        }
    }}
    testcase add_zero {
        let a = ir.add_input(IrType::new(Size::variable(), DType::F32, ));
        ir.add_unary(a, Unary::BinaryWithConst { op: Binary::Add, val: 0.0.into(), lhs: false })?
    },
    testcase sub_zero {
        let a = ir.add_input(IrType::new(Size::variable(), DType::F32));
        ir.add_unary(a, Unary::BinaryWithConst { op: Binary::Sub, val: 0.0.into(), lhs: true })?
    }
}

foldrule! {
    rulename FoldSubFromZero on ir
    rewrites (unary = [IrUnary] (a))
    into [{
        let val = match a.ty().dtype() {
            DType::F32 => (-1.0).into(),
            DType::I32 => (-1).into()
        };
        let op = Unary::BinaryWithConst { op: Binary::Mul, val, lhs: true };
        IrUnary::new(a.ty(), op)?
    }] (a)
    given {{
        if let Unary::BinaryWithConst { op: Binary::Sub, val, lhs: false } = unary.op() {
            val == 0.0.into() || val == 0.into()
        } else {
            false
        }
    }}
}

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
