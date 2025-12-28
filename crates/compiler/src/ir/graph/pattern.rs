#[macro_export]
macro_rules! if_find_and_bind_pattern {
    (@sub $($inner:tt)*) => { 1usize };
    (@len $(($($inner:tt)*))*) => {
        [$(if_find_and_bind_pattern!(@sub $($inner:tt)*)),*].len()
    };
    {
        @input ($count:expr) $op:ident ($irname:ident, {$($then:tt)*})
    } => {
        {$($then)*}
    };
    {
        @input ($count:expr) $op:ident ($irname:ident, {$($then:tt)*})
        ($innername:ident = [$innertype:ty] $($inner:tt)*) $($tail:tt)*
    } => {
        let __op_id = $irname.get_parent_op($op.inputs()[$count])?;
        let $innername = $irname.get_op(__op_id)?;

        if_find_and_bind_pattern! {
            @input ($count + 1usize) $op ($irname, {
                if_find_and_bind_pattern! {
                    @operation ($irname, {$($then)*})
                    $innername = [$innertype] $($inner)*
                }
            })
            $($tail)*
        }
    };
    {
        @input ($count:expr) $op:ident ($irname:ident, {$($then:tt)*})
        ($name:ident) $($tail:tt)*
    } => {
        let $name = $irname.get_node($op.inputs()[$count])?;

        if_find_and_bind_pattern! {
            @input ($count + 1usize) $op ($irname, {$($then)*})
            $($tail)*
        }
    };
    {
        @operation ($irname:ident, {$($then:tt)*})
        $op:ident = [$optype:ty] $($tail:tt)*
    } => {
        if IR::downcast::<$optype>($op.op()).is_some()
            && $op.inputs().len() == if_find_and_bind_pattern!(@len $($tail)*)
        {
            if_find_and_bind_pattern! {
                @input (0usize) $op ($irname, {
                    let $op = IR::downcast::<$optype>($op.op()).unwrap();
                    $($then)*
                })
                $($tail)*
            }
        }
    };
    {
        target: $irname:ident, $op:expr,
        pattern: ($opname:ident $($tail:tt)*)
        then: {
            $($then:tt)*
        }
    } => {
        let $opname = $op;
        if_find_and_bind_pattern! {
            @operation ($irname, {$($then)*})
            $opname $($tail)*
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{
        core::{Binary, DType, Size},
        ir::{
            IR, IRTrace,
            graph::{IrType, operation::IrBinary},
        },
    };

    #[test]
    fn find_basic_pattern() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let ty = IrType::new(Size::variable(), DType::F32);
        let a = ir.add_input(ty);
        let b = ir.add_input(ty);
        let c = ir.add_binary(a, b, Binary::Add)?;
        let d = ir.add_binary(c, b, Binary::Sub)?;

        ir.register_output(d);

        let mut found = 0;

        for op in ir.operations() {
            if_find_and_bind_pattern! {
                target: ir, op,
                pattern: (binary = [IrBinary] (a) (b))
                then: {
                    if binary.op() == Binary::Add
                        && ir.is_input(a.id())?
                        && ir.is_input(b.id())?
                    {
                        found += 1
                    }
                }
            };
        }

        assert_eq!(found, 1);

        ir.check_valid()
    }

    #[test]
    fn find_nested_pattern_lhs() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let ty = IrType::new(Size::variable(), DType::F32);
        let a = ir.add_input(ty);
        let b = ir.add_input(ty);
        let c = ir.add_binary(a, b, Binary::Add)?;
        let d = ir.add_binary(c, b, Binary::Sub)?;

        ir.register_output(d);

        let mut found = 0;

        for op in ir.operations() {
            if_find_and_bind_pattern! {
                target: ir, op,
                pattern: (
                    binary1 = [IrBinary]
                        (binary2 = [IrBinary] (_a) (b))
                        (c)
                )
                then: {
                    if binary1.op() == Binary::Sub && binary2.op() == Binary::Add && b.id() == c.id()
                    {
                        found += 1
                    }
                }
            };
        }

        assert_eq!(found, 1);

        ir.check_valid()
    }

    #[test]
    fn find_nested_pattern_rhs() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let ty = IrType::new(Size::variable(), DType::F32);
        let a = ir.add_input(ty);
        let b = ir.add_input(ty);
        let c = ir.add_binary(a, b, Binary::Add)?;
        let d = ir.add_binary(b, c, Binary::Sub)?;

        ir.register_output(d);

        let mut found = 0;

        for op in ir.operations() {
            if_find_and_bind_pattern! {
                target: ir, op,
                pattern: (
                    binary1 = [IrBinary]
                        (a)
                        (binary2 = [IrBinary] (_b) (c))
                )
                then: {
                    if binary1.op() == Binary::Sub && binary2.op() == Binary::Add && a.id() == c.id()
                    {
                        found += 1
                    }
                }
            };
        }

        assert_eq!(found, 1);

        ir.check_valid()
    }
}
