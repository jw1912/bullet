mod buffer;
mod error;
mod tests;

use crate::{graph::operation::Activation, shape::Shape};

pub use buffer::DeviceBuffer;
pub use error::OperationError;

type OperationResult<T> = Result<(), OperationError<T>>;

#[allow(clippy::too_many_arguments)]
pub trait Device: Sized + 'static {
    type IdType;
    type DeviceError: std::fmt::Debug;
    type BufferI32: DeviceBuffer<Self, i32>;
    type BufferF32: DeviceBuffer<Self, f32>;

    fn new(id: Self::IdType) -> Result<Self, Self::DeviceError>;

    fn synchronise(&self) -> Result<(), Self::DeviceError>;

    fn get_last_device_error(&self) -> Result<(), Self::DeviceError>;

    fn activate(
        size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
        activation: Activation,
    ) -> OperationResult<Self::DeviceError>;

    fn backprop_activate(
        size: usize,
        input: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
        output_grad: &Self::BufferF32,
        activation: Activation,
    ) -> OperationResult<Self::DeviceError>;

    fn sgemm(
        input_a: &Self::BufferF32,
        shape_a: Shape,
        trans_a: bool,
        input_b: &Self::BufferF32,
        shape_b: Shape,
        trans_b: bool,
        output: &mut Self::BufferF32,
        increment: bool,
    ) -> OperationResult<Self::DeviceError>;

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
    ) -> OperationResult<Self::DeviceError>;

    fn add_assign_single_to_batched_scaled(
        single_size: usize,
        batch_size: usize,
        ones: &Self::BufferF32,
        alpha: f32,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn reduce_add(
        ones: &Self::BufferF32,
        size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

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
    ) -> OperationResult<Self::DeviceError>;

    fn sparse_affine(
        batch_size: usize,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

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
    ) -> OperationResult<Self::DeviceError>;

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
    ) -> OperationResult<Self::DeviceError>;

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
    ) -> OperationResult<Self::DeviceError>;

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
    ) -> OperationResult<Self::DeviceError>;

    fn mask(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        inputs: &Self::BufferF32,
        masks: &Self::BufferI32,
        outputs: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn backprop_mask(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        output_grads: &Self::BufferF32,
        masks: &Self::BufferI32,
        input_grads: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn pairwise(
        single_size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
        post_concat: bool,
    ) -> OperationResult<Self::DeviceError>;

    fn backprop_pairwise(
        single_size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
        post_concat: bool,
    ) -> OperationResult<Self::DeviceError>;

    fn select(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn select_backprop(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn gather(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        inputs: &Self::BufferF32,
        indices: &Self::BufferI32,
        outputs: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn backprop_gather(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        output_grads: &Self::BufferF32,
        indices: &Self::BufferI32,
        input_grads: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn abs_power_error(
        power: f32,
        size: usize,
        input_a: &Self::BufferF32,
        input_b: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn backprop_abs_power_error_single(
        power: f32,
        size: usize,
        input_a: &Self::BufferF32,
        input_b: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_a_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn softmax_across_batch_masked(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        masks: &Self::BufferI32,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn crossentropy_masked(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        masks: &Self::BufferI32,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
        error: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn backprop_softmax_crossentropy_masked(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        masks: &Self::BufferI32,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn adam(
        size: usize,
        params: &mut Self::BufferF32,
        gradient: &Self::BufferF32,
        momentum: &mut Self::BufferF32,
        velocity: &mut Self::BufferF32,
        beta1: f32,
        beta2: f32,
        gradient_factor: f32,
        learning_rate: f32,
        denom: bool,
    ) -> OperationResult<Self::DeviceError>;

    fn clip(size: usize, params: &mut Self::BufferF32, min: f32, max: f32) -> OperationResult<Self::DeviceError>;

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;
}
