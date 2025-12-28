use std::{fmt, rc::Rc};

use crate::{
    core::{Binary, DTypeTensor, DTypeValue, Unary},
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
    }

    pub fn add_fold(mut self, fold: impl Fold) -> Self {
        self.0.push(Rc::new(fold));
        self
    }

    fn apply_single_fold(&self, ir: &mut IR, mut id: IrOperationId) -> Result<bool, IRTrace> {
        let mut success = false;

        'fold: loop {
            let op = ir.get_op(id)?;

            if let Some(consts) = fold_constant_expression(ir, op.inputs(), op.op())? {
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

fn fold_constant_expression(
    ir: &IR,
    inputs: &[IrNodeId],
    op: &Rc<dyn IrOperationType>,
) -> Result<Option<Vec<Constant>>, IRTrace> {
    if !inputs.is_empty() {
        let mut consts = Vec::new();

        for &input in inputs {
            let parent = ir.get_op(ir.get_parent_op(input)?)?;

            if let Some(Constant(value)) = IrOperation::downcast(parent.op()).cloned() {
                consts.push(value);
            } else if let Some(Some(scalar)) = IrOperation::downcast(parent.op()).map(ScalarConstant::to_tensor) {
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

macro_rules! foldrule_matching {
    ($matching:pat = $cond:expr ;;; $inner:expr) => {
        if let $matching = $cond {
            $inner
        }
    };
    ($cond:expr ;;; $inner:expr) => {
        if $cond {
            $inner
        }
    };
}

macro_rules! foldrule {
    {
        rulename $name:ident on $irname:ident
        rewrites ($($input:ident),*) -> $old_op:pat,
        into ($($output:ident),*) -> $new_op:expr,
        iff $($cond:tt)*
    } => {
        foldrule! {
            $name, $irname,
            ($($input),*), Some($old_op),
            ($($output),*), $new_op,
            $($cond)*
        }
    };
    {
        rulename $name:ident on $irname:ident
        rewrites ($($input:ident),*) -> $old_opname:ident = $old_ty:ty,
        into ($($output:ident),*) -> $new_op:expr,
        iff $($cond:tt)*
    } => {
        foldrule! {
            $name, $irname,
            ($($input),*), Some::<&$old_ty>($old_opname),
            ($($output),*), $new_op,
            $($cond)*
        }
    };
    {
        $name:ident, $irname:ident,
        ($($input:ident),*), $old_op:pat,
        ($($output:ident),*), $new_op:expr,
        $($cond:tt)*
    } => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;

        impl Fold for $name {
            fn fold(
                &self,
                $irname: &IR,
                inputs: &[IrNodeId],
                operation: &Rc<dyn IrOperationType>,
            ) -> Result<Option<AddOperation>, IRTrace> {
                if let [$($input),*] = inputs[..] {
                    if let $old_op = IrOperation::downcast(operation) {
                        $(let $input = $irname.get_node($input)?;)*

                        foldrule_matching!($($cond)* ;;; {
                            let new_op = $new_op;
                            let new_inputs = vec![$($output.id()),*];
                            return Ok(Some(AddOperation(new_inputs, Ok(Rc::new(new_op)))));
                        });
                    }
                }

                Ok(None)
            }
        }
    };
}

foldrule! {
    rulename FoldFixedSizeScalarConst on _ir
    rewrites () -> Constant(value),
    into () -> ScalarConstant(scalar, value.size().into()),
    iff Some(scalar) = value.scalar()
}

foldrule! {
    rulename FoldVarSizeScalarConst on ir
    rewrites (input) -> broadcast = BroadcastAcrossDimension,
    into () -> ScalarConstant(*val, broadcast.outputs()[0].size()),
    iff Some(ScalarConstant(val, _)) = ir.is_child_of(input.id())?
}

foldrule! {
    rulename FoldScalarConstLhsIntoBinary on ir
    rewrites (a, b) -> binary = IrBinary,
    into (b) -> {
        let ty = IrType::new(size, val.dtype());
        let op = Unary::BinaryWithConst { op: binary.op(), val, lhs: false };
        IrUnary::new(ty, op)?
    },
    iff Some(ScalarConstant(val, size)) = ir.is_child_of(a.id())?.cloned()
}

foldrule! {
    rulename FoldScalarConstRhsIntoBinary on ir
    rewrites (a, b) -> binary = IrBinary,
    into (a) -> {
        let ty = IrType::new(size, val.dtype());
        let op = Unary::BinaryWithConst { op: binary.op(), val, lhs: true };
        IrUnary::new(ty, op).unwrap()
    },
    iff Some(ScalarConstant(val, size)) = ir.is_child_of(b.id())?.cloned()
}

foldrule! {
    rulename FoldScalarConstIntoUnary on ir
    rewrites (a) -> unary = IrUnary,
    into () -> ScalarConstant(unary.op().evaluate(val).unwrap(), size),
    iff Some(ScalarConstant(val, size)) = ir.is_child_of(a.id())?.cloned()
}

foldrule! {
    rulename FoldNegLhsIntoAdd on ir
    rewrites (a, b) -> binary = IrBinary,
    into (b, a) -> IrBinary::new(b.ty(), a.ty(), Binary::Sub)?,
    iff {
        if binary.op() == Binary::Add
            && let Some(unary) = ir.is_child_of::<IrUnary>(a.id())?
            && let Unary::BinaryWithConst { val, op: Binary::Mul, .. } = unary.op()
            && (val == DTypeValue::F32(-1.0) || val == DTypeValue::I32(-1))
        {
            true
        } else {
            false
        }
    }
}

foldrule! {
    rulename FoldNegRhsIntoAdd on ir
    rewrites (a, b) -> binary = IrBinary,
    into (a, b) -> IrBinary::new(a.ty(), b.ty(), Binary::Sub)?,
    iff {
        if binary.op() == Binary::Add
            && let Some(unary) = ir.is_child_of::<IrUnary>(b.id())?
            && let Unary::BinaryWithConst { val, op: Binary::Mul, .. } = unary.op()
            && (val == DTypeValue::F32(-1.0) || val == DTypeValue::I32(-1))
        {
            true
        } else {
            false
        }
    }
}

foldrule! {
    rulename FoldXMinusXIntoZero on ir
    rewrites (a, b) -> binary = IrBinary,
    into () -> {
        let ty = a.ty();
        ScalarConstant(DTypeValue::zero(ty.dtype()), ty.size())
    },
    iff binary.op() == Binary::Sub && a.id() == b.id()
}

foldrule! {
    rulename FoldMulByOne on ir
    rewrites (a) -> unary = IrUnary,
    into (a) -> IrCopy(a.ty()),
    iff {
        if let Unary::BinaryWithConst { op: Binary::Mul, val, .. } = unary.op() {
            val == 1.0.into() || val == 1.into()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{DType, DTypeValue, Size};

    use super::*;

    fn test_scalar_const(size: Size, fold: FoldPass) -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let broadcast1 = BroadcastAcrossDimension::new(DType::I32, [1], 0, size);
        let broadcast2 = BroadcastAcrossDimension::new(DType::I32, [size], 0, size);
        let a = ir.add_const(DTypeValue::I32(1).into());
        let b = ir.add_op([a], broadcast1)?[0];
        let c = ir.add_op([b], broadcast2)?[0];

        ir.register_output(c);
        ir.transform(fold)?;

        let expected = ScalarConstant(DTypeValue::I32(1), size * size);
        assert_eq!(ir.is_child_of::<ScalarConstant>(c)?, Some(&expected));

        ir.check_valid()
    }

    #[test]
    fn const_size_scalar_const() -> Result<(), IRTrace> {
        test_scalar_const(16.into(), FoldPass::from(FoldFixedSizeScalarConst))
    }

    #[test]
    fn var_size_scalar_const() -> Result<(), IRTrace> {
        let fold = FoldPass::from(FoldFixedSizeScalarConst).add_fold(FoldVarSizeScalarConst);
        test_scalar_const(Size::variable(), fold)
    }

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
        assert_eq!(ir.is_child_of(d)?, Some(&IrUnary::new(ty, op)?));
        assert_eq!(ir.is_child_of(e)?, Some(&ScalarConstant(0.0.into(), size)));

        ir.check_valid()
    }
}
