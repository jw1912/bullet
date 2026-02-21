#[macro_export]
macro_rules! if_find_and_bind_pattern {
    // no more inputs to check, so can paste in success code
    {
        @input ($count:expr) ($irname:ident, {$($then:tt)*})
    } => {
        {$($then)*}
    };
    // munch op inputs
    {
        @input ($count:expr) ($irname:ident, {$($then:tt)*})
        ($innername:ident = [$innertype:ty] $($inner:tt)*) $($tail:tt)*
    } => {
        let __op_id = $irname.get_parent_op($innername)?;
        let $innername = $irname.get_op(__op_id)?;

        $crate::if_find_and_bind_pattern! {
            @input ($count + 1usize) ($irname, {
                $crate::if_find_and_bind_pattern! {
                    @operation ($irname, {$($then)*})
                    $innername = [$innertype] $($inner)*
                }
            })
            $($tail)*
        }
    };
    // munch leaf inputs
    {
        @input ($count:expr) ($irname:ident, {$($then:tt)*})
        ($name:ident) $($tail:tt)*
    } => {
        let $name = $irname.get_node($name)?;

        $crate::if_find_and_bind_pattern! {
            @input ($count + 1usize) ($irname, {$($then)*})
            $($tail)*
        }
    };
    // check operation and number of inputs match, binding relevant idents
    {
        @operation ($irname:ident, {$($then:tt)*})
        $op:ident = [$optype:ty] $(($childname:ident $($tail:tt)*))*
    } => {
        if let [$($childname),*] = $op.inputs()[..]
            && let Some($op) = $crate::tensor::TensorOp::downcast::<$optype>($op.data())
        {
            $crate::if_find_and_bind_pattern! {
                @input (0usize) ($irname, {$($then)*})
                $(($childname $($tail)*))*
            }
        }
    };
    // top level invocation
    ($irname:ident, $op:expr, ($opname:ident $($tail:tt)*), $($then:tt)*) => {
        let $opname = $op;
        $crate::if_find_and_bind_pattern! {
            @operation ($irname, {$($then)*})
            $opname $($tail)*
        }
    };
}
