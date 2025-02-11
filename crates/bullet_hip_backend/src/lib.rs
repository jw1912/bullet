mod backend;
pub mod dense;
mod matmul;
pub mod sparse;

#[cfg(test)]
mod tests;

pub use backend::ExecutionContext;
use backend::{util, Buffer};

use bullet_core::{device::Device, graph::operation::Activation, shape::Shape, tensor};

pub type DenseMatrix = tensor::DenseMatrix<ExecutionContext>;
pub type SparseMatrix = tensor::SparseMatrix<ExecutionContext>;
pub type Matrix = tensor::Matrix<ExecutionContext>;
pub type Tensor = tensor::Tensor<ExecutionContext>;

impl Device for ExecutionContext {
    type BufferF32 = Buffer<f32>;
    type BufferI32 = Buffer<i32>;
    type IdType = ();

    fn new(_: Self::IdType) -> Self {
        Self::default()
    }

    fn synchronise(&self) {
        util::device_synchronise();
    }

    fn panic_if_device_error(&self, msg: &str) {
        util::panic_if_device_error(msg);
    }

    fn activate(size: usize, input: &Self::BufferF32, output: &mut Self::BufferF32, activation: Activation) {
        match activation {
            Activation::Identity => panic!("No-op!"),
            Activation::ReLU => dense::relu(size, input, output),
            Activation::CReLU => dense::crelu(size, input, output),
            Activation::SCReLU => dense::screlu(size, input, output),
            Activation::SqrReLU => dense::sqrrelu(size, input, output),
            Activation::Sigmoid => dense::sigmoid(size, input, output),
        }
    }

    fn backprop_activate(
        size: usize,
        input: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
        output_grad: &Self::BufferF32,
        activation: Activation,
    ) {
        match activation {
            Activation::Identity => panic!("No-op!"),
            Activation::ReLU => dense::relu_backward(size, input, input_grad, output_grad),
            Activation::CReLU => dense::crelu_backward(size, input, input_grad, output_grad),
            Activation::SCReLU => dense::screlu_backward(size, input, input_grad, output_grad),
            Activation::SqrReLU => dense::sqrrelu_backward(size, input, input_grad, output_grad),
            Activation::Sigmoid => dense::sigmoid_backward(size, input, input_grad, output_grad),
        }
    }

    fn add_assign_single_to_batched_scaled(
        single_size: usize,
        batch_size: usize,
        ones: &Self::BufferF32,
        alpha: f32,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) {
        dense::add_assign_single_to_batched_scaled(single_size, batch_size, ones, alpha, input, output);
    }

    fn sgemm(
        input_a: &Self::BufferF32,
        shape_a: Shape,
        trans_a: bool,
        input_b: &Self::BufferF32,
        shape_b: Shape,
        trans_b: bool,
        output: &mut Self::BufferF32,
        increment: bool,
    ) {
        matmul::sgemm(input_a, shape_a, trans_a, input_b, shape_b, trans_b, output, increment);
    }

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
    ) {
        matmul::sgemm_batched(batch_size, input_a, shape_a, trans_a, input_b, shape_b, trans_b, output, increment);
    }

    fn backprop_abs_power_error_single(
        power: f32,
        size: usize,
        input_a: &Self::BufferF32,
        input_b: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_a_grad: &mut Self::BufferF32,
    ) {
        dense::backprop_abs_power_error_single(power, size, input_a, input_b, output_grad, input_a_grad);
    }

    fn backprop_pairwise(
        single_size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
        post_concat: bool,
    ) {
        dense::backprop_pairwise(single_size, batch_size, input, output_grad, input_grad, post_concat);
    }

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
    ) {
        sparse::backprop_sparse_affine(
            batch_size,
            input_a,
            input_a_grad,
            shape_a,
            input_b,
            shape_b,
            nnz,
            input_c,
            input_c_grad,
            outputs,
            output_grad,
        );
    }

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
    ) {
        sparse::backprop_sparse_affine_dual_activate(
            batch_size,
            input_a,
            input_a_grad,
            shape_a,
            input_b1,
            input_b2,
            shape_b,
            nnz,
            input_c,
            input_c_grad,
            outputs,
            output_grad,
            activation,
        );
    }

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
    ) {
        dense::copy_or_add_strided(
            rows,
            cols,
            input,
            input_offset,
            input_stride,
            output,
            output_offset,
            output_stride,
            add,
        );
    }

    fn pairwise(
        single_size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
        post_concat: bool,
    ) {
        dense::pairwise(single_size, batch_size, input, output, post_concat);
    }

    fn abs_power_error(
        power: f32,
        size: usize,
        input_a: &Self::BufferF32,
        input_b: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) {
        dense::abs_power_error(power, size, input_a, input_b, output);
    }

    fn sparse_affine(
        batch_size: usize,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        output: &mut Self::BufferF32,
    ) {
        sparse::sparse_affine(batch_size, input_a, shape_a, input_b, shape_b, nnz, input_c, output);
    }

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
    ) {
        sparse::sparse_affine_dual_activate(
            batch_size, input_a, shape_a, input_b1, input_b2, shape_b, nnz, input_c, output, activation,
        );
    }

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
    ) {
        dense::adamw(
            size,
            params,
            gradient,
            momentum,
            velocity,
            beta1,
            beta2,
            min_weight,
            max_weight,
            decay,
            gradient_factor,
            learning_rate,
        );
    }

    fn linear_comb_single(
        size: usize,
        alpha: f32,
        input_a: Option<&Self::BufferF32>,
        beta: f32,
        input_b: Option<&Self::BufferF32>,
        output: &mut Self::BufferF32,
    ) {
        dense::linear_comb_single(size, alpha, input_a, beta, input_b, output);
    }

    fn reduce_add(
        ones: &Self::BufferF32,
        size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) {
        dense::reduce_add(ones, size, batch_size, input, output);
    }

    fn select(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) {
        sparse::select(batch_size, input_size, output_size, input, indices, output);
    }

    fn select_backprop(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) {
        sparse::select_backprop(batch_size, input_size, output_size, indices, output_grad, input_grad);
    }

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) {
        sparse::sparse_to_dense(batch_size, size, nnz, sparse, dense);
    }

    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) {
        dense::softmax_across_batch(batch_size, single_size, input, output);
    }

    fn crossentropy(size: usize, pred: &Self::BufferF32, target: &Self::BufferF32, output: &mut Self::BufferF32) {
        dense::crossentropy(size, pred, target, output);
    }

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) {
        dense::backprop_softmax_crossentropy(size, softmaxed, target, output_grad, input_grad);
    }

    fn mask(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        inputs: &Self::BufferF32,
        masks: &Self::BufferI32,
        outputs: &mut Self::BufferF32,
    ) {
        sparse::mask(batch_size, single_size, inputs, masks, nnz, outputs);
    }

    fn backprop_mask(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        output_grads: &Self::BufferF32,
        masks: &Self::BufferI32,
        input_grads: &mut Self::BufferF32,
    ) {
        sparse::backprop_mask(batch_size, single_size, output_grads, masks, nnz, input_grads);
    }

    fn gather(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        inputs: &Self::BufferF32,
        indices: &Self::BufferI32,
        outputs: &mut Self::BufferF32,
    ) {
        sparse::gather(batch_size, input_size, output_size, inputs, indices, outputs);
    }

    fn backprop_gather(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        output_grads: &Self::BufferF32,
        indices: &Self::BufferI32,
        input_grads: &mut Self::BufferF32,
    ) {
        sparse::backprop_gather(batch_size, input_size, output_size, output_grads, indices, input_grads);
    }

    fn softmax_across_batch_masked(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        masks: &Self::BufferI32,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) {
        sparse::softmax_across_batch_masked(batch_size, single_size, nnz, masks, input, output);
    }

    fn crossentropy_masked(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        masks: &Self::BufferI32,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
        error: &mut Self::BufferF32,
    ) {
        sparse::crossentropy_masked(batch_size, single_size, nnz, masks, pred, target, output, error);
    }

    fn backprop_softmax_crossentropy_masked(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        masks: &Self::BufferI32,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) {
        sparse::backprop_softmax_crossentropy_masked(
            batch_size,
            single_size,
            nnz,
            masks,
            softmaxed,
            target,
            output_grad,
            input_grad,
        );
    }
}
