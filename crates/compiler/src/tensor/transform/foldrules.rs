use std::{fmt, rc::Rc};

use crate::{
    ir::Op,
    tensor::{
        DValue, IRTrace, Tensor, TensorIR,
        operation::{BroadcastAcrossDimension, CABinary, CABinaryOp, Constant, CopyOp, ScalarConstant, UnaryOp},
        transform::modify::AddOperation,
    },
};

#[cfg(test)]
use crate::tensor::{Size, TValue};

/// A fold rule is a special case for the simplest form of rewrite, that being an
/// entirely local transform on an operation, changing only the operation itself
/// by replacing it with a new operation that has equivalent outputs.
///
/// An example is if `Y = a * (b * X)`, we can replace this with `Y = (a * b) * X`.
/// If the child `b * X` is now unused then it can be eliminated as dead code later
/// on and we have saved a full tensor operation, but otherwise we have lost nothing
/// by performing this transformation.
pub trait FoldRule: fmt::Debug + 'static {
    fn fold(&self, ir: &TensorIR, operation: &Op<Tensor>) -> Result<Option<AddOperation>, IRTrace>;
}

macro_rules! foldrule {
    (@maybe_matching ($inner:expr) ($($matching:pat = $cond:expr;)+)) => {
        if $(let $matching = $cond)&&+ {
            $inner
        }
    };
    (@maybe_matching ($inner:expr) ($cond:expr) ) => {
        if $cond {
            $inner
        }
    };
    (@maybe_matching ($inner:expr) ()) => {
        $inner
    };
    {
        rulename $name:ident on $irname:ident
        rewrites ($($pattern:tt)*)
        into [$new_op:expr] $(($output:ident))*
        $(given {
            $($cond:tt)*
        })?
        $(testcase $testname:ident $testcase:expr),*
    } => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;

        impl FoldRule for $name {
            fn fold(
                &self,
                #[allow(unused)]
                $irname: &TensorIR,
                operation: &Op<Tensor>,
            ) -> Result<Option<AddOperation>, IRTrace> {
                $crate::if_find_and_bind_pattern!(
                    $irname,
                    operation,
                    ($($pattern)*),
                    foldrule! {
                        @maybe_matching
                        ({
                            let new_op = $new_op;
                            let new_inputs = vec![$($output.id()),*];
                            return Ok(Some(AddOperation::new(new_inputs, Ok(Rc::new(new_op)))));
                        })
                        ($($($cond)*)?)
                    }
                );

                Ok(None)
            }
        }

        $(
        #[cfg(test)]
        #[test]
        fn $testname() -> Result<(), IRTrace> {
            let mut $irname = TensorIR::default();

            let output = {
                $testcase
            };

            $irname.register_output(output);
            let op = $irname.get_op($irname.get_parent_op(output)?)?;
            assert!($name.fold(&$irname, op)?.is_some());

            $irname.check_valid()
        }
        )*
    };
}

foldrule! {
    rulename EvalScalarConstUnary on ir
    rewrites (unary = [UnaryOp] (scalar = [ScalarConstant]))
    into [ScalarConstant(unary.op().evaluate(scalar.0).unwrap(), scalar.1)]
}

foldrule! {
    rulename EvalScalarConstBinary on ir
    rewrites (binary = [CABinaryOp] (lhs = [ScalarConstant]) (rhs = [ScalarConstant]))
    into [ScalarConstant(binary.op().evaluate(lhs.0, rhs.0).unwrap(), lhs.1)]
}

foldrule! {
    rulename FoldFixedSizeScalarConst on ir
    rewrites (constant = [Constant])
    into [ScalarConstant(scalar, constant.0.size().into())]
    given {
        Some(scalar) = constant.0.scalar();
    }
    testcase fixed_size_scalar_const {
        ir.add_const(TValue::I32(vec![1; 16]))
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
    rulename FoldConstIdentities on ir
    rewrites (binary = [CABinaryOp] (scalar = [ScalarConstant]) (a))
    into [CopyOp(a.ty())] (a)
    given {{
        let ScalarConstant(val, _) = *scalar;
        match binary.op() {
            CABinary::Mul => val == 1.0.into() || val == 1.into(),
            CABinary::Add => val == 0.0.into() || val == 0.into(),
            CABinary::Max => val == f32::MIN.into() || val == i32::MIN.into(),
            CABinary::Min => val == f32::MAX.into() || val == i32::MAX.into(),
        }
    }}
}

foldrule! {
    rulename FoldMulByZero on ir
    rewrites (binary = [CABinaryOp] (scalar = [ScalarConstant]) (a))
    into [ScalarConstant(DValue::zero(a.ty().dtype()), a.ty().size())]
    given {
        binary.op() == CABinary::Mul && (scalar.0 == 0.0.into() || scalar.0 == 0.into())
    }
}
