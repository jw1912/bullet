mod blas;
mod buffer;
mod matmul;

use std::sync::Arc;

use buffer::CudaBuffer;
use bullet_core::{
    device::Device,
    graph::operation::{Activation, ConvolutionDescription},
    shape::Shape,
    tensor,
};

use cudarc::{cublas::CudaBlas, driver::CudaDevice};

pub type DenseMatrix = tensor::DenseMatrix<ExecutionContext>;
pub type SparseMatrix = tensor::SparseMatrix<ExecutionContext>;
pub type Matrix = tensor::Matrix<ExecutionContext>;
pub type Tensor = tensor::Tensor<ExecutionContext>;

pub struct ExecutionContext {
    device: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
}

#[allow(unused)]
impl Device for ExecutionContext {
    type BufferF32 = CudaBuffer<f32>;
    type BufferI32 = CudaBuffer<i32>;
    type IdType = usize;

    fn new(id: Self::IdType) -> Self {
        let device = CudaDevice::new(id).unwrap();
        let blas = Arc::new(CudaBlas::new(device.clone()).unwrap());

        Self { device, blas }
    }

    fn synchronise(&self) {
        self.device.synchronize().unwrap();
    }

    // using `cudarc` we handle all errors at time of occurrence
    fn panic_if_device_error(&self, _: &str) {}

    fn activate(size: usize, input: &Self::BufferF32, output: &mut Self::BufferF32, activation: Activation) {
        unimplemented!()
    }

    fn backprop_activate(
        size: usize,
        input: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
        output_grad: &Self::BufferF32,
        activation: Activation,
    ) {
        unimplemented!()
    }

    fn sgemm(
        input_a: &DenseMatrix,
        shape_a: Shape,
        trans_a: bool,
        input_b: &DenseMatrix,
        shape_b: Shape,
        trans_b: bool,
        output: &mut DenseMatrix,
        output_shape: Shape,
        increment: bool,
    ) {
        matmul::sgemm(input_a, shape_a, trans_a, input_b, shape_b, trans_b, output, output_shape, increment);
    }

    fn sgemm_batched(
        input_a: &DenseMatrix,
        trans_a: bool,
        input_b: &DenseMatrix,
        trans_b: bool,
        output: &mut DenseMatrix,
        increment: bool,
    ) {
        matmul::sgemm_batched(input_a, trans_a, input_b, trans_b, output, increment);
    }

    fn sgemm_batched_reshaped(
        input_a: &DenseMatrix,
        shape_a: Shape,
        trans_a: bool,
        input_b: &DenseMatrix,
        shape_b: Shape,
        trans_b: bool,
        output: &mut DenseMatrix,
        increment: bool,
    ) {
        unimplemented!()
    }

    fn add_assign_single_to_batched_scaled(
        single_size: usize,
        batch_size: usize,
        ones: &Self::BufferF32,
        alpha: f32,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) {
        unimplemented!()
    }

    fn linear_comb(
        ones: &Self::BufferF32,
        alpha: f32,
        input_a: &DenseMatrix,
        beta: f32,
        input_b: &DenseMatrix,
        output: &mut DenseMatrix,
    ) {
        unimplemented!()
    }

    fn backprop_add_single_scaled(
        ones: &Self::BufferF32,
        alpha: f32,
        input: &DenseMatrix,
        input_grad: &mut DenseMatrix,
        output_grad: &DenseMatrix,
    ) {
        unimplemented!()
    }

    fn reduce_add_batch(ones: &Self::BufferF32, input: &DenseMatrix, output: &mut DenseMatrix) {
        unimplemented!()
    }

    fn sparse_affine(
        input_a: &DenseMatrix,
        input_b: &SparseMatrix,
        input_c: Option<&DenseMatrix>,
        output: &mut DenseMatrix,
    ) {
        unimplemented!()
    }

    fn backprop_sparse_affine(
        input_a: &DenseMatrix,
        input_a_grad: &mut DenseMatrix,
        input_b: &SparseMatrix,
        input_c: Option<&DenseMatrix>,
        input_c_grad: Option<&mut DenseMatrix>,
        outputs: &DenseMatrix,
        output_grad: &DenseMatrix,
    ) {
        unimplemented!()
    }

    fn sparse_affine_dual_activate(
        input_a: &DenseMatrix,
        input_b1: &SparseMatrix,
        input_b2: &SparseMatrix,
        input_c: &DenseMatrix,
        output: &mut DenseMatrix,
        activation: Activation,
    ) {
        unimplemented!()
    }

    fn backprop_sparse_affine_dual_activate(
        input_a: &DenseMatrix,
        input_a_grad: &mut DenseMatrix,
        input_b1: &SparseMatrix,
        input_b2: &SparseMatrix,
        input_c: &DenseMatrix,
        input_c_grad: &mut DenseMatrix,
        outputs: &DenseMatrix,
        output_grad: &DenseMatrix,
        activation: Activation,
    ) {
        unimplemented!()
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
        unimplemented!()
    }

    fn mask(inputs: &DenseMatrix, masks: &SparseMatrix, outputs: &mut DenseMatrix) {
        unimplemented!()
    }

    fn backprop_mask(output_grads: &DenseMatrix, masks: &SparseMatrix, input_grads: &mut DenseMatrix) {
        unimplemented!()
    }

    fn pairwise(input: &DenseMatrix, output: &mut DenseMatrix, post_concat: bool) {
        unimplemented!()
    }

    fn backprop_pairwise(
        input: &DenseMatrix,
        output_grad: &DenseMatrix,
        input_grad: &mut DenseMatrix,
        post_concat: bool,
    ) {
        unimplemented!()
    }

    fn select(input: &DenseMatrix, indices: &SparseMatrix, output: &mut DenseMatrix) {
        unimplemented!()
    }

    fn select_backprop(
        input: &DenseMatrix,
        indices: &SparseMatrix,
        output_grad: &DenseMatrix,
        input_grad: &mut DenseMatrix,
    ) {
        unimplemented!()
    }

    fn gather(inputs: &DenseMatrix, indices: &SparseMatrix, outputs: &mut DenseMatrix) {
        unimplemented!()
    }

    fn backprop_gather(
        output_grads: &DenseMatrix,
        indices: &SparseMatrix,
        inputs: &DenseMatrix,
        input_grads: &mut DenseMatrix,
    ) {
        unimplemented!()
    }

    fn power_error(power: f32, input_a: &DenseMatrix, input_b: &DenseMatrix, output: &mut DenseMatrix) {
        unimplemented!()
    }

    fn backprop_abs_power_error_single(
        power: f32,
        input_a: &DenseMatrix,
        input_b: &DenseMatrix,
        output_grad: &DenseMatrix,
        input_a_grad: &mut DenseMatrix,
    ) {
        unimplemented!()
    }

    fn convolution_forward(
        desc: &ConvolutionDescription,
        filters: &DenseMatrix,
        input: &DenseMatrix,
        output: &mut DenseMatrix,
    ) {
        unimplemented!()
    }

    // could maybe split into separate filter and input grads?
    fn convolution_backward(
        desc: &ConvolutionDescription,
        filters: &DenseMatrix,
        filters_grad: Option<&mut DenseMatrix>,
        input: &DenseMatrix,
        input_grad: Option<&mut DenseMatrix>,
        output_grad: &DenseMatrix,
    ) {
        unimplemented!()
    }

    fn softmax_across_batch(input: &DenseMatrix, output: &mut DenseMatrix) {
        unimplemented!()
    }

    fn crossentropy_loss(
        ones: &Self::BufferF32,
        softmaxed: &DenseMatrix,
        target: &DenseMatrix,
        individual_losses: &mut DenseMatrix,
        output: &mut DenseMatrix,
    ) {
        unimplemented!()
    }

    fn backprop_softmax_crossentropy_loss(
        softmaxed: &DenseMatrix,
        target: &DenseMatrix,
        output_grad: &DenseMatrix,
        input_grad: &mut DenseMatrix,
    ) {
        unimplemented!()
    }

    fn softmax_across_batch_masked(mask: &SparseMatrix, input: &DenseMatrix, output: &mut DenseMatrix) {
        unimplemented!()
    }

    fn crossentropy_loss_masked(
        mask: &SparseMatrix,
        softmaxed: &DenseMatrix,
        target: &DenseMatrix,
        individual_losses: &mut DenseMatrix,
        output: &mut DenseMatrix,
    ) {
        unimplemented!()
    }

    fn backprop_softmax_crossentropy_loss_masked(
        mask: &SparseMatrix,
        softmaxed: &DenseMatrix,
        target: &DenseMatrix,
        output_grad: &DenseMatrix,
        input_grad: &mut DenseMatrix,
    ) {
        unimplemented!()
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
        unimplemented!()
    }
}
