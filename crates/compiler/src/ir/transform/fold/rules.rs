#[macro_export]
macro_rules! foldrule {
    {
        rulename $visible:vis $name:ident on $irname:ident
        rewrites ($($input:ident),*) -> $old_op:pat,
        into ($($output:ident),*) -> $new_op:expr,
        iff {
            $($cond:tt)*
        }
        $(testcase $testname:ident |$ir:ident| $testcase:expr)?
    } => {
        foldrule! {
            $visible $name, $irname,
            ($($input),*), Some($old_op),
            ($($output),*), $new_op,
            {$($cond)*}
            $($testname, $ir, $testcase)?
        }
    };
    {
        rulename $visible:vis $name:ident on $irname:ident
        rewrites ($($input:ident),*) -> $old_opname:ident = $old_ty:ty,
        into ($($output:ident),*) -> $new_op:expr,
        iff {
            $($cond:tt)*
        }
        $(testcase $testname:ident |$ir:ident| $testcase:expr)?
    } => {
        foldrule! {
            $visible $name, $irname,
            ($($input),*), Some::<&$old_ty>($old_opname),
            ($($output),*), $new_op,
            {$($cond)*}
            $($testname, $ir, $testcase)?
        }
    };
    {
        $visible:vis $name:ident, $irname:ident,
        ($($input:ident),*), $old_op:pat,
        ($($output:ident),*), $new_op:expr,
        {$($cond:tt)*}
        $($testname:ident, $ir:ident, $testcase:expr)?
    } => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        $visible struct $name;

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

                        foldrule!(matching $($cond)* ;;; {
                            let new_op = $new_op;
                            let new_inputs = vec![$($output.id()),*];
                            return Ok(Some(AddOperation(new_inputs, Ok(Rc::new(new_op)))));
                        });
                    }
                }

                Ok(None)
            }
        }

        $(
        #[cfg(test)]
        #[test]
        fn $testname() -> Result<(), IRTrace> {
            let mut $ir = IR::default();

            let output = {
                $testcase
            };

            $ir.register_output(output);

            let pass = FoldPass::from($name);
            let op = $ir.get_parent_op(output)?;

            assert!(pass.apply_single_fold(&mut $ir, op)?);

            $ir.check_valid()
        }
        )?
    };
    (matching $matching:pat = $cond:expr ;;; $inner:expr) => {
        if let $matching = $cond {
            $inner
        }
    };
    (matching $cond:expr ;;; $inner:expr) => {
        if $cond {
            $inner
        }
    };
}

pub(crate) use foldrule;
