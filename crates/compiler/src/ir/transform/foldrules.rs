use std::{fmt, rc::Rc};

use crate::{
    core::{Binary, DTypeValue, Unary},
    ir::{
        IR, IRTrace,
        graph::{IrError, IrOperation, IrType},
        operation::{BroadcastAcrossDimension, Constant, IrBinary, IrCopy, IrUnary, ScalarConstant},
        transform::modify::AddOperation,
    },
};

#[cfg(test)]
use crate::core::{DType, DTypeTensor, Size};

pub trait FoldRule: fmt::Debug + 'static {
    fn fold(&self, ir: &IR, operation: &IrOperation) -> Result<Option<AddOperation>, IRTrace>;
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
                $irname: &IR,
                operation: &IrOperation,
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
                            return Ok(Some(AddOperation(new_inputs, Ok(Rc::new(new_op)))));
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
            let mut $irname = IR::default();

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
    rulename FoldUnaryUnaryIntoUnary on ir
    rewrites (u1 = [IrUnary] (u2 = [IrUnary] (input)))
    into [IrUnary::new(input.ty(), Unary::BinaryWithConst { op, val, lhs: true })?] (input)
    given {
        Some((op, val)) = {
            if let (
                Unary::BinaryWithConst { op: op1, val: val1, .. },
                Unary::BinaryWithConst { op: op2, val: val2, .. },
            ) = (u1.op(), u2.op()) {
                match (op1, op2) {
                    (Binary::Mul, Binary::Mul) => Some((
                        Binary::Mul,
                        Binary::Mul.evaluate(val1, val2)
                            .ok_or::<IrError>(format!("Unable to eval {val1} * {val2}").into())?
                    )),
                    (Binary::Add, Binary::Add) => Some((
                        Binary::Add,
                        Binary::Add.evaluate(val1, val2)
                            .ok_or::<IrError>(format!("Unable to eval {val1} * {val2}").into())?
                    )),
                    _ => None,
                }
            } else {
                None
            }
        };
    }
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
                Binary::DivByI32 => lhs && val == 1.into(),
                _ => false,
            }
        } else {
            false
        }
    }}
    testcase add_zero {
        let a = ir.add_input(IrType::new(Size::variable(), DType::F32, ));
        ir.add_unary(a, Unary::BinaryWithConst { op: Binary::Add, val: 0.0.into(), lhs: false })?
    }
}

foldrule! {
    rulename FoldMulByZero on ir
    rewrites (unary = [IrUnary] (a))
    into [ScalarConstant(DTypeValue::zero(a.ty().dtype()), a.ty().size())]
    given {{
        if let Unary::BinaryWithConst { op: Binary::Mul, val, .. } = unary.op() {
            val == 0.0.into() || val == 0.into()
        } else {
            false
        }
    }}
}

foldrule! {
    rulename FoldXAddMulX on ir
    rewrites (binary = [IrBinary] (a) (unary = [IrUnary] (b)))
    into [IrUnary::new(a.ty(), Unary::BinaryWithConst { op: Binary::Mul, val, lhs: true })?] (a)
    given {
        Some(val) = {
            if a.id() == b.id() && binary.op() == Binary::Add
                && let Unary::BinaryWithConst { op: Binary::Mul, val, .. } = unary.op()
            {
                Binary::Add.evaluate(val, DTypeValue::one(val.dtype()))
            } else {
                None
            }
        };
    }
}

foldrule! {
    rulename FoldMulXAddMulX on ir
    rewrites (binary = [IrBinary] (u1 = [IrUnary] (a)) (u2 = [IrUnary] (b)))
    into [IrUnary::new(a.ty(), Unary::BinaryWithConst { op: Binary::Mul, val, lhs: true })?] (a)
    given {
        Some(val) = {
            if a.id() == b.id() && binary.op() == Binary::Add
                && let Unary::BinaryWithConst { op: Binary::Mul, val: val1, .. } = u1.op()
                && let Unary::BinaryWithConst { op: Binary::Mul, val: val2, .. } = u2.op()
            {
                Binary::Mul.evaluate(val1, val2)
            } else {
                None
            }
        };
    }
}
