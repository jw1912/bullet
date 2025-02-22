use bullet_core::graph::tests;

use crate::ExecutionContext;

tests::make_tests! {
    ExecutionContext::default(),
    matmul,
    sparse_affine,
    sparse_affine_dual,
    check_not_batched,
    relu,
    crelu,
    screlu,
    sqrrelu,
    concat,
}
