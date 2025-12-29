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
        rulename $visible:vis $name:ident on $irname:ident
        rewrites ($($pattern:tt)*)
        into [$new_op:expr] $(($output:ident))*
        $(given {
            $($cond:tt)*
        })?
        $(testcase $testname:ident $testcase:expr),*
    } => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        $visible struct $name;

        impl Fold for $name {
            fn fold(
                &self,
                #[allow(unused)]
                $irname: &IR,
                operation: &IrOperation,
            ) -> Result<Option<AddOperation>, IRTrace> {
                $crate::if_find_and_bind_pattern! {
                    target: $irname, operation,
                    pattern: ($($pattern)*)
                    then: {
                        foldrule! {
                            @maybe_matching
                            ({
                                let new_op = $new_op;
                                let new_inputs = vec![$($output.id()),*];
                                return Ok(Some(AddOperation(new_inputs, Ok(Rc::new(new_op)))));
                            })
                            ($($($cond)*)?)
                        }
                    }
                }

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

            let pass = FoldPass::from($name);
            let op = $irname.get_parent_op(output)?;

            assert!(pass.apply_single_fold(&mut $irname, op)?);

            $irname.check_valid()
        }
        )*
    };
}
