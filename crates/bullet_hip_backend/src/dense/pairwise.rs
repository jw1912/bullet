use crate::backend::{ops, Buffer};

pub fn pairwise(
    mut single_size: usize,
    mut batch_size: usize,
    input: &Buffer<f32>,
    output: &mut Buffer<f32>,
    post_concat: bool,
) {
    if post_concat {
        assert_eq!(single_size % 2, 0);
        single_size /= 2;
        batch_size *= 2;
    }

    assert_eq!(single_size % 2, 0);

    unsafe {
        ops::pairwiseMul(batch_size, single_size / 2, input.ptr(), output.mut_ptr());
    }
}

pub fn backprop_pairwise(
    mut single_size: usize,
    mut batch_size: usize,
    input: &Buffer<f32>,
    output_grad: &Buffer<f32>,
    input_grad: &mut Buffer<f32>,
    post_concat: bool,
) {
    if post_concat {
        assert_eq!(single_size % 2, 0);
        single_size /= 2;
        batch_size *= 2;
    }

    assert_eq!(single_size % 2, 0);

    unsafe {
        ops::backpropPairwiseMul(batch_size, single_size / 2, input.ptr(), output_grad.ptr(), input_grad.mut_ptr());
    }
}
