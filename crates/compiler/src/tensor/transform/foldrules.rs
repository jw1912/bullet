use std::fmt;

use crate::{
    ir::Op,
    tensor::{
        DValue, IRTrace, Tensor, TensorIR,
        operation::{
            CABinary, CABinaryOp, Constant, CopyOp, Power, ScalarConstant, SliceAcrossDimension, SparseMatmul,
            SparseMatmulBwd, SparseMatmulBwdMulti, Unary, UnaryOp,
            autograd::{CReLU, CustomAutogradOp, DiffableFromOutputOp, ReLU, SCReLU, Sigmoid, SqrReLU},
        },
        transform::modify::AddOperation,
    },
};

#[cfg(test)]
use crate::tensor::TValue;

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

#[macro_export]
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
                            return Ok(Some(AddOperation::new(new_inputs, Ok($crate::tensor::TensorOp::new(new_op)))));
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
    rulename FoldConstToScalarConst on ir
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

foldrule! {
    rulename FoldSparseMatmulBwdToMulti on ir
    rewrites (bwd = [SparseMatmulBwd] (a) (b))
    into [SparseMatmulBwdMulti::new(*bwd)] (a) (b)
}

foldrule! {
    rulename FoldPowBy1 on ir
    rewrites (_b = [Power] (a) (scalar = [ScalarConstant]))
    into [CopyOp(a.ty())] (a)
    given {
        scalar.0 == 1.0.into()
    }
}

foldrule! {
    rulename FoldPowBy2 on ir
    rewrites (_b = [Power] (a) (scalar = [ScalarConstant]))
    into [CABinaryOp::new(a.ty(), CABinary::Mul)] (a) (a)
    given {
        scalar.0 == 2.0.into() || scalar.0 == 2.into()
    }
}

foldrule! {
    rulename FoldAbsSquared on ir
    rewrites (_c = [CABinaryOp] (a = [UnaryOp] (i1)) (b = [UnaryOp] (i2)))
    into [CABinaryOp::new(i1.ty(), CABinary::Mul)] (i1) (i1)
    given {
        a.op() == Unary::Abs && b.op() == Unary::Abs && i1.id() == i2.id()
    }
}

foldrule! {
    rulename PeepholeReLU on ir
    rewrites (relu = [CABinaryOp] (scalar = [ScalarConstant]) (input))
    into [{
        let ty = input.ty();
        let op = DiffableFromOutputOp(ReLU, ty.dtype(), ty.size());
        CustomAutogradOp::new(op)?
    }] (input)
    given {
        relu.op() == CABinary::Max && scalar.0 == DValue::zero(input.ty().dtype())
    }
}

foldrule! {
    rulename PeepholeCReLU on ir
    rewrites (crelu = [CABinaryOp] (scalar = [ScalarConstant]) (relu = [CustomAutogradOp] (input)))
    into [{
        let ty = input.ty();
        let op = DiffableFromOutputOp(CReLU, ty.dtype(), ty.size());
        CustomAutogradOp::new(op)?
    }] (input)
    given {
        relu.downcast::<DiffableFromOutputOp<ReLU>>().is_some() && crelu.op() == CABinary::Min && scalar.0 == DValue::one(input.ty().dtype())
    }
}

foldrule! {
    rulename PeepholeSCReLU on ir
    rewrites (screlu = [CABinaryOp] (crelu1 = [CustomAutogradOp] (input1)) (crelu2 = [CustomAutogradOp] (input2)))
    into [{
        let ty = input1.ty();
        let op = DiffableFromOutputOp(SCReLU, ty.dtype(), ty.size());
        CustomAutogradOp::new(op)?
    }] (input1)
    given {
        {
            let op1 = crelu1.downcast::<DiffableFromOutputOp<CReLU>>();
            let op2 = crelu2.downcast::<DiffableFromOutputOp<CReLU>>();
            op1.is_some() && op1 == op2 && input1.id() == input2.id() && screlu.op() == CABinary::Mul
        }
    }
}

foldrule! {
    rulename PeepholeSqrReLU on ir
    rewrites (screlu = [CABinaryOp] (relu1 = [CustomAutogradOp] (input1)) (relu2 = [CustomAutogradOp] (input2)))
    into [{
        let ty = input1.ty();
        let op = DiffableFromOutputOp(SqrReLU, ty.dtype(), ty.size());
        CustomAutogradOp::new(op)?
    }] (input1)
    given {
        {
            let op1 = relu1.downcast::<DiffableFromOutputOp<ReLU>>();
            let op2 = relu2.downcast::<DiffableFromOutputOp<ReLU>>();
            op1.is_some() && op1 == op2 && input1.id() == input2.id() && screlu.op() == CABinary::Mul
        }
    }
}

foldrule! {
    rulename PeepholeSigmoid on ir
    rewrites (
        sigmoid = [UnaryOp]
            (denom = [CABinaryOp]
                (one = [ScalarConstant])
                (exp = [UnaryOp] (neg = [CABinaryOp] (neg_one = [ScalarConstant]) (input)))))
    into [{
        let ty = input.ty();
        let op = DiffableFromOutputOp(Sigmoid, ty.dtype(), ty.size());
        CustomAutogradOp::new(op)?
    }] (input)
    given {
        sigmoid.op() == Unary::Reciprocal
            && denom.op() == CABinary::Add
            && one.0 == 1.0.into()
            && exp.op() == Unary::Exp
            && neg.op() == CABinary::Mul
            && neg_one.0 == (-1.0).into()
    }
}

foldrule! {
    rulename FoldSlicedSparseMatmul on ir
    rewrites (spmm = [SparseMatmul] (sliced = [SliceAcrossDimension] (weights)) (inputs))
    into [SparseMatmul::new(spmm.dtype(), spmm.batch(), spmm.rows(), spmm.cols(), sliced.dimen(), sliced.start(), spmm.nnz())?] (weights) (inputs)
    given {
        spmm.offset() == 0 && spmm.rows() == spmm.stride() && sliced.inner().get() == 1
    }
}
