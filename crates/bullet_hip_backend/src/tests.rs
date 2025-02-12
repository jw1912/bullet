use bullet_core::graph::tests;

use crate::ExecutionContext;

tests::make_tests! {
    ExecutionContext::default(),
    matmul,
    sparse_affine,
    sparse_affine_dual,
    relu,
    crelu,
    screlu,
    sqrrelu,
    concat,
}
