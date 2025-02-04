use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

use crate::{
    graph::operation::{Activation, ConvolutionDescription},
    shape::Shape,
    tensor::{DenseMatrix, SparseMatrix},
};

#[cfg(feature = "cuda")]
/// # Safety
/// Must be representable on all devices and valid
/// as zeroed bits
pub unsafe trait ValidType: DeviceRepr + ValidAsZeroBits {}

#[cfg(not(feature = "cuda"))]
/// # Safety
/// Must be representable on all devices and valid
/// as zeroed bits
pub unsafe trait ValidType {}

unsafe impl ValidType for f32 {}
unsafe impl ValidType for i32 {}

pub trait DeviceBuffer<D: Device, T: ValidType> {
    fn new(device: Arc<D>, size: usize) -> Self;

    fn size(&self) -> usize;

    fn device(&self) -> Arc<D>;

    fn set_zero(&mut self);

    fn load_from_device(&mut self, buf: &Self, bytes: usize);

    fn load_from_slice(&mut self, buf: &[T]);

    fn write_into_slice(&self, buf: &mut [T], bytes: usize);
}

#[allow(clippy::too_many_arguments)]
pub trait Device: Sized + 'static {
    type IdType;
    type Buffer<T: ValidType>: DeviceBuffer<Self, T>;

    fn new(id: Self::IdType) -> Self;

    fn synchronise(&self);

    fn panic_if_device_error(&self, msg: &str);

    fn activate(input: &DenseMatrix<Self>, output: &mut DenseMatrix<Self>, activation: Activation);

    fn sgemm(
        input_a: &DenseMatrix<Self>,
        shape_a: Shape,
        trans_a: bool,
        input_b: &DenseMatrix<Self>,
        shape_b: Shape,
        trans_b: bool,
        output: &mut DenseMatrix<Self>,
        output_shape: Shape,
        increment: bool,
    );

    fn sgemm_batched(
        input_a: &DenseMatrix<Self>,
        trans_a: bool,
        input_b: &DenseMatrix<Self>,
        trans_b: bool,
        output: &mut DenseMatrix<Self>,
        increment: bool,
    );

    fn sgemm_batched_reshaped(
        input_a: &DenseMatrix<Self>,
        shape_a: Shape,
        trans_a: bool,
        input_b: &DenseMatrix<Self>,
        shape_b: Shape,
        trans_b: bool,
        output: &mut DenseMatrix<Self>,
        increment: bool,
    );

    fn add_assign_single_to_batched_scaled(
        ones: &Self::Buffer<f32>,
        alpha: f32,
        input: &DenseMatrix<Self>,
        output: &mut DenseMatrix<Self>,
    );

    fn linear_comb(
        ones: &Self::Buffer<f32>,
        alpha: f32,
        input_a: &DenseMatrix<Self>,
        beta: f32,
        input_b: &DenseMatrix<Self>,
        output: &mut DenseMatrix<Self>,
    );

    fn backprop_add_single_scaled(
        ones: &Self::Buffer<f32>,
        alpha: f32,
        input: &DenseMatrix<Self>,
        input_grad: &mut DenseMatrix<Self>,
        output_grad: &DenseMatrix<Self>,
    );

    fn reduce_add_batch(ones: &Self::Buffer<f32>, input: &DenseMatrix<Self>, output: &mut DenseMatrix<Self>);

    fn sparse_affine(
        input_a: &DenseMatrix<Self>,
        input_b: &SparseMatrix<Self>,
        input_c: Option<&DenseMatrix<Self>>,
        output: &mut DenseMatrix<Self>,
    );

    fn backprop_sparse_affine(
        input_a: &DenseMatrix<Self>,
        input_a_grad: &mut DenseMatrix<Self>,
        input_b: &SparseMatrix<Self>,
        input_c: Option<&DenseMatrix<Self>>,
        input_c_grad: Option<&mut DenseMatrix<Self>>,
        outputs: &DenseMatrix<Self>,
        output_grad: &DenseMatrix<Self>,
    );

    fn sparse_affine_dual_activate(
        input_a: &DenseMatrix<Self>,
        input_b1: &SparseMatrix<Self>,
        input_b2: &SparseMatrix<Self>,
        input_c: &DenseMatrix<Self>,
        output: &mut DenseMatrix<Self>,
        activation: Activation,
    );

    fn backprop_sparse_affine_dual_activate(
        input_a: &DenseMatrix<Self>,
        input_a_grad: &mut DenseMatrix<Self>,
        input_b1: &SparseMatrix<Self>,
        input_b2: &SparseMatrix<Self>,
        input_c: &DenseMatrix<Self>,
        input_c_grad: &mut DenseMatrix<Self>,
        outputs: &DenseMatrix<Self>,
        output_grad: &DenseMatrix<Self>,
        activation: Activation,
    );

    fn copy_or_add_strided(
        rows: usize,
        cols: usize,
        input: &Self::Buffer<f32>,
        input_offset: usize,
        input_stride: usize,
        output: &mut Self::Buffer<f32>,
        output_offset: usize,
        output_stride: usize,
        add: bool,
    );

    fn mask(inputs: &DenseMatrix<Self>, masks: &SparseMatrix<Self>, outputs: &mut DenseMatrix<Self>);

    fn backprop_mask(output_grads: &DenseMatrix<Self>, masks: &SparseMatrix<Self>, input_grads: &mut DenseMatrix<Self>);

    fn pairwise(input: &DenseMatrix<Self>, output: &mut DenseMatrix<Self>, post_concat: bool);

    fn backprop_pairwise(
        input: &DenseMatrix<Self>,
        output_grad: &DenseMatrix<Self>,
        input_grad: &mut DenseMatrix<Self>,
        post_concat: bool,
    );

    fn select(input: &DenseMatrix<Self>, indices: &SparseMatrix<Self>, output: &mut DenseMatrix<Self>);

    fn select_backprop(
        input: &DenseMatrix<Self>,
        indices: &SparseMatrix<Self>,
        output_grad: &DenseMatrix<Self>,
        input_grad: &mut DenseMatrix<Self>,
    );

    fn slice_vector_batched(input: &DenseMatrix<Self>, start: usize, end: usize, output: &mut DenseMatrix<Self>);

    fn backprop_slice_vector_batched(
        input: &DenseMatrix<Self>,
        input_grad: &mut DenseMatrix<Self>,
        start: usize,
        end: usize,
        output_grad: &DenseMatrix<Self>,
    );

    fn gather(inputs: &DenseMatrix<Self>, indices: &SparseMatrix<Self>, outputs: &mut DenseMatrix<Self>);

    fn backprop_gather(
        output_grads: &DenseMatrix<Self>,
        indices: &SparseMatrix<Self>,
        inputs: &DenseMatrix<Self>,
        input_grads: &mut DenseMatrix<Self>,
    );

    fn power_error(
        power: f32,
        input_a: &DenseMatrix<Self>,
        input_b: &DenseMatrix<Self>,
        output: &mut DenseMatrix<Self>,
    );

    fn backprop_abs_power_error_single(
        power: f32,
        input_a: &DenseMatrix<Self>,
        input_b: &DenseMatrix<Self>,
        output_grad: &DenseMatrix<Self>,
        input_a_grad: &mut DenseMatrix<Self>,
    );

    fn convolution_forward(
        desc: &ConvolutionDescription,
        filters: &DenseMatrix<Self>,
        input: &DenseMatrix<Self>,
        output: &mut DenseMatrix<Self>,
    );

    // could maybe split into separate filter and input grads?
    fn convolution_backward(
        desc: &ConvolutionDescription,
        filters: &DenseMatrix<Self>,
        filters_grad: Option<&mut DenseMatrix<Self>>,
        input: &DenseMatrix<Self>,
        input_grad: Option<&mut DenseMatrix<Self>>,
        output_grad: &DenseMatrix<Self>,
    );

    fn softmax_across_batch(input: &DenseMatrix<Self>, output: &mut DenseMatrix<Self>);

    fn crossentropy_loss(
        ones: &Self::Buffer<f32>,
        softmaxed: &DenseMatrix<Self>,
        target: &DenseMatrix<Self>,
        individual_losses: &mut DenseMatrix<Self>,
        output: &mut DenseMatrix<Self>,
    );

    fn backprop_softmax_crossentropy_loss(
        softmaxed: &DenseMatrix<Self>,
        target: &DenseMatrix<Self>,
        output_grad: &DenseMatrix<Self>,
        input_grad: &mut DenseMatrix<Self>,
    );

    fn softmax_across_batch_masked(
        mask: &SparseMatrix<Self>,
        input: &DenseMatrix<Self>,
        output: &mut DenseMatrix<Self>,
    );

    fn crossentropy_loss_masked(
        mask: &SparseMatrix<Self>,
        softmaxed: &DenseMatrix<Self>,
        target: &DenseMatrix<Self>,
        individual_losses: &mut DenseMatrix<Self>,
        output: &mut DenseMatrix<Self>,
    );

    fn backprop_softmax_crossentropy_loss_masked(
        mask: &SparseMatrix<Self>,
        softmaxed: &DenseMatrix<Self>,
        target: &DenseMatrix<Self>,
        output_grad: &DenseMatrix<Self>,
        input_grad: &mut DenseMatrix<Self>,
    );
}
