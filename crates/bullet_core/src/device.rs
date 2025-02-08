mod buffer;
mod tests;

use crate::{graph::operation::Activation, shape::Shape};

pub use buffer::DeviceBuffer;

#[allow(clippy::too_many_arguments)]
pub trait Device: Sized + 'static {
    type IdType;
    type BufferI32: DeviceBuffer<Self, i32>;
    type BufferF32: DeviceBuffer<Self, f32>;

    fn new(id: Self::IdType) -> Self;

    fn synchronise(&self);

    fn panic_if_device_error(&self, msg: &str);

    fn activate(size: usize, input: &Self::BufferF32, output: &mut Self::BufferF32, activation: Activation);

    fn backprop_activate(
        size: usize,
        input: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
        output_grad: &Self::BufferF32,
        activation: Activation,
    );

    fn sgemm(
        input_a: &Self::BufferF32,
        shape_a: Shape,
        trans_a: bool,
        input_b: &Self::BufferF32,
        shape_b: Shape,
        trans_b: bool,
        output: &mut Self::BufferF32,
        increment: bool,
    );

    fn sgemm_batched(
        batch_size: usize,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        trans_a: bool,
        input_b: &Self::BufferF32,
        shape_b: Shape,
        trans_b: bool,
        output: &mut Self::BufferF32,
        increment: bool,
    );

    fn add_assign_single_to_batched_scaled(
        single_size: usize,
        batch_size: usize,
        ones: &Self::BufferF32,
        alpha: f32,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    );

    fn reduce_add(
        ones: &Self::BufferF32,
        size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    );

    /// If `input_a = None`, then take `input_a = output`, i.e. perform the
    /// in place operation `output = alpha * output + beta * input_b`.
    ///
    /// If `input_b = None` then this is equivalent to a scaling operation.
    fn linear_comb_single(
        size: usize,
        alpha: f32,
        input_a: Option<&Self::BufferF32>,
        beta: f32,
        input_b: Option<&Self::BufferF32>,
        output: &mut Self::BufferF32,
    );

    fn sparse_affine(
        batch_size: usize,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        output: &mut Self::BufferF32,
    );

    fn backprop_sparse_affine(
        batch_size: usize,
        input_a: &Self::BufferF32,
        input_a_grad: &mut Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        input_c_grad: Option<&mut Self::BufferF32>,
        outputs: &Self::BufferF32,
        output_grad: &Self::BufferF32,
    );

    fn sparse_affine_dual_activate(
        batch_size: usize,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        input_b1: &Self::BufferI32,
        input_b2: &Self::BufferI32,
        shape_b: Shape,
        nnz: usize,
        input_c: &Self::BufferF32,
        output: &mut Self::BufferF32,
        activation: Activation,
    );

    fn backprop_sparse_affine_dual_activate(
        batch_size: usize,
        input_a: &Self::BufferF32,
        input_a_grad: &mut Self::BufferF32,
        shape_a: Shape,
        input_b1: &Self::BufferI32,
        input_b2: &Self::BufferI32,
        shape_b: Shape,
        nnz: usize,
        input_c: &Self::BufferF32,
        input_c_grad: &mut Self::BufferF32,
        outputs: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        activation: Activation,
    );

    fn copy_or_add_strided(
        rows: usize,
        cols: usize,
        input: &Self::BufferF32,
        input_offset: usize,
        input_stride: usize,
        output: &mut Self::BufferF32,
        output_offset: usize,
        output_stride: usize,
        add: bool,
    );

    //fn mask(inputs: &DenseMatrix<Self>, masks: &SparseMatrix<Self>, outputs: &mut DenseMatrix<Self>);

    //fn backprop_mask(output_grads: &DenseMatrix<Self>, masks: &SparseMatrix<Self>, input_grads: &mut DenseMatrix<Self>);

    fn pairwise(
        single_size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
        post_concat: bool,
    );

    fn backprop_pairwise(
        single_size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
        post_concat: bool,
    );

    fn select(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    );

    fn select_backprop(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    );

    //fn gather(inputs: &DenseMatrix<Self>, indices: &SparseMatrix<Self>, outputs: &mut DenseMatrix<Self>);

    //fn backprop_gather(
    //    output_grads: &DenseMatrix<Self>,
    //    indices: &SparseMatrix<Self>,
    //    inputs: &DenseMatrix<Self>,
    //    input_grads: &mut DenseMatrix<Self>,
    //);

    fn abs_power_error(
        power: f32,
        size: usize,
        input_a: &Self::BufferF32,
        input_b: &Self::BufferF32,
        output: &mut Self::BufferF32,
    );

    fn backprop_abs_power_error_single(
        power: f32,
        size: usize,
        input_a: &Self::BufferF32,
        input_b: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_a_grad: &mut Self::BufferF32,
    );

    //fn softmax_across_batch(input: &DenseMatrix<Self>, output: &mut DenseMatrix<Self>);

    //fn crossentropy_loss(
    //    ones: &Self::BufferF32,
    //    softmaxed: &DenseMatrix<Self>,
    //    target: &DenseMatrix<Self>,
    //    individual_losses: &mut DenseMatrix<Self>,
    //    output: &mut DenseMatrix<Self>,
    //);

    //fn backprop_softmax_crossentropy_loss(
    //    softmaxed: &DenseMatrix<Self>,
    //    target: &DenseMatrix<Self>,
    //    output_grad: &DenseMatrix<Self>,
    //    input_grad: &mut DenseMatrix<Self>,
    //);

    //fn softmax_across_batch_masked(
    //    mask: &SparseMatrix<Self>,
    //    input: &DenseMatrix<Self>,
    //    output: &mut DenseMatrix<Self>,
    //);

    //fn crossentropy_loss_masked(
    //    mask: &SparseMatrix<Self>,
    //    softmaxed: &DenseMatrix<Self>,
    //    target: &DenseMatrix<Self>,
    //    individual_losses: &mut DenseMatrix<Self>,
    //    output: &mut DenseMatrix<Self>,
    //);

    //fn backprop_softmax_crossentropy_loss_masked(
    //    mask: &SparseMatrix<Self>,
    //    softmaxed: &DenseMatrix<Self>,
    //    target: &DenseMatrix<Self>,
    //    output_grad: &DenseMatrix<Self>,
    //    input_grad: &mut DenseMatrix<Self>,
    //);

    fn adamw(
        size: usize,
        params: &mut Self::BufferF32,
        gradient: &Self::BufferF32,
        momentum: &mut Self::BufferF32,
        velocity: &mut Self::BufferF32,
        beta1: f32,
        beta2: f32,
        min_weight: f32,
        max_weight: f32,
        decay: f32,
        gradient_factor: f32,
        learning_rate: f32,
    );

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    );
}
