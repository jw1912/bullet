use bullet_core::graph::tests;

use crate::ExecutionContext;

tests::make_tests! {
    ExecutionContext::default(),
    matmul,
    matmul2,
    sparse_affine,
    sparse_affine_batched_biases,
    sparse_affine_dual,
    sparse_affine_check_not_batched,
    relu,
    crelu,
    screlu,
    sqrrelu,
    concat,
}
