/// Macro for finding operation patterns rooted at one operation in IR.
///
/// Usage:
///
/// `if_find_and_bind_pattern!(ir: IR | IrGraph, operation: IrOperation, pattern, success_block)`
///
/// If the `pattern` is found in `ir` starting rooted at `operation`, then `success_block`
/// is ran. Any identifiers appearing in `pattern` are appropriately bound.
///
/// Simple example:
/// ```
/// # use bullet_compiler::{
/// #     core::{CABinary, DType, Size},
/// #     if_find_and_bind_pattern,
/// #     ir::{
/// #         graph::{IrGraph, IrError, IrType},
/// #         operation::CABinaryOp,
/// #   },
/// # };
/// #
/// # let mut ir = IrGraph::default();
/// # let ty = IrType::new(Size::variable(), DType::F32);
/// # let node_a = ir.add_input(ty);
/// # let node_b = ir.add_input(ty);
/// let node_c = ir.add_op([node_a, node_b], CABinaryOp::new(ty, CABinary::Add))?[0];
/// let target_op = ir.get_op(ir.get_parent_op(node_c)?)?;
///
/// let mut found = false;
/// if_find_and_bind_pattern!(
///     ir,
///     target_op,
///     (binary = [CABinaryOp] (a) (b)),
///     found = binary.op() == CABinary::Add && a.id() == node_a && b.id() == node_b
/// );
///
/// assert!(found);
/// # ir.check_valid()?;
/// # Ok::<(), IrError>(())
/// ```
///
/// This can be extended to more complex nested patterns:
/// ```
/// # use bullet_compiler::{
/// #     core::{CABinary, DType, Size},
/// #     if_find_and_bind_pattern,
/// #     ir::{
/// #         graph::{IrGraph, IrError, IrType, IrInput},
/// #         operation::CABinaryOp,
/// #     },
/// # };
/// #
/// # let mut ir = IrGraph::default();
/// # let ty = IrType::new(Size::variable(), DType::F32);
/// let node_a = ir.add_input(ty);
/// let node_b = ir.add_input(ty);
/// let node_c = ir.add_op([node_a, node_b], CABinaryOp::new(ty, CABinary::Add))?[0];
/// let node_d = ir.add_op([node_c, node_b], CABinaryOp::new(ty, CABinary::Min))?[0];
/// let node_e = ir.add_op([node_c, node_d], CABinaryOp::new(ty, CABinary::Mul))?[0];
/// let target_op = ir.get_op(ir.get_parent_op(node_e)?)?;
///
/// let mut found = false;
/// if_find_and_bind_pattern!(
///     ir,
///     target_op,
///     (b_e = [CABinaryOp]
///         (b_c = [CABinaryOp] (a1 = [IrInput]) (b1 = [IrInput]))
///         (b_d = [CABinaryOp]
///             (b_c2 = [CABinaryOp] (a2 = [IrInput]) (b2 = [IrInput]))
///             (b3 = [IrInput])
///         )
///     ),
///     found = b_c == b_c2
///         && b_c.op() == CABinary::Add
///         && b_d.op() == CABinary::Min
///         && b_e.op() == CABinary::Mul
/// );
///
/// assert!(found);
/// # ir.check_valid()?;
/// # Ok::<(), IrError>(())
/// ```
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
            && let Some($op) = $op.downcast::<$optype>()
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
