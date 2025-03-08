mod backend;
pub mod dense;
mod matmul;
pub mod sparse;

#[cfg(test)]
mod tests;

pub use backend::ExecutionContext;
use backend::{bindings, util, Buffer};

use bullet_core::backend::{
    device::{
        base::{Activation, AdamConfig, BaseOperations},
        blas::{BlasOperations, GemmConfig, Shape},
        Device, OperationError,
    },
    tensor,
};

pub type DenseMatrix = tensor::DenseMatrix<ExecutionContext>;
pub type SparseMatrix = tensor::SparseMatrix<ExecutionContext>;
pub type Matrix = tensor::Matrix<ExecutionContext>;
pub type Tensor = tensor::Tensor<ExecutionContext>;

#[derive(Debug)]
pub enum DeviceError {
    Cuda(bindings::cudaError_t),
    Cublas(bindings::cublasStatus_t),
    ExpectedIllegalAddressAccess,
}

impl From<bindings::cublasStatus_t> for Result<(), DeviceError> {
    fn from(value: bindings::cublasStatus_t) -> Self {
        if value == bindings::CUBLAS_SUCCESS {
            Ok(())
        } else {
            Err(DeviceError::Cublas(value))
        }
    }
}

impl From<bindings::cudaError_t> for Result<(), DeviceError> {
    fn from(value: bindings::cudaError_t) -> Self {
        if value == bindings::SUCCESS {
            Ok(())
        } else {
            Err(DeviceError::Cuda(value))
        }
    }
}

pub(crate) type OperationResult = Result<(), OperationError<DeviceError>>;

impl BlasOperations for Buffer<f32> {
    type BlasError = DeviceError;

    fn gemm(&mut self, config: &GemmConfig, a: &Self, b: &Self) -> Result<(), Self::BlasError> {
        matmul::sgemm(config, a, b, self)
    }

    fn gebmm(&mut self, config: &GemmConfig, batch_size: usize, a: &Self, b: &Self) -> Result<(), Self::BlasError> {
        matmul::sgemm_batched(config, batch_size, a, b, self)
    }

    fn geam(
        &mut self,
        size: usize,
        alpha: f32,
        a: Option<&Self>,
        beta: f32,
        b: Option<&Self>,
    ) -> Result<(), Self::BlasError> {
        dense::linear_comb_single(size, alpha, a, beta, b, self)
    }
}

impl BaseOperations for Buffer<f32> {
    type BaseError = DeviceError;

    fn activate_fwd(&mut self, size: usize, a: &Self, act: Activation) -> Result<(), Self::BaseError> {
        match act {
            Activation::Identity => panic!("No-op!"),
            Activation::ReLU => dense::relu(size, a, self),
            Activation::CReLU => dense::crelu(size, a, self),
            Activation::SCReLU => dense::screlu(size, a, self),
            Activation::SqrReLU => dense::sqrrelu(size, a, self),
            Activation::Sigmoid => dense::sigmoid(size, a, self),
            Activation::Square => dense::square(size, a, self),
        }
    }

    fn activate_bwd(&mut self, size: usize, a: &Self, grd: &Self, act: Activation) -> Result<(), Self::BaseError> {
        match act {
            Activation::Identity => panic!("No-op!"),
            Activation::ReLU => dense::relu_backward(size, a, self, grd),
            Activation::CReLU => dense::crelu_backward(size, a, self, grd),
            Activation::SCReLU => dense::screlu_backward(size, a, self, grd),
            Activation::SqrReLU => dense::sqrrelu_backward(size, a, self, grd),
            Activation::Sigmoid => dense::sigmoid_backward(size, a, self, grd),
            Activation::Square => dense::square_backward(size, a, self, grd),
        }
    }

    fn copy_or_add_strided(
        &mut self,
        add: bool,
        rows: usize,
        cols: usize,
        offset: usize,
        stride: usize,
        a: &Self,
        offset_a: usize,
        stride_a: usize,
    ) -> Result<(), Self::BaseError> {
        dense::copy_or_add_strided(rows, cols, a, offset_a, stride_a, self, offset, stride, add)
    }

    fn pairwise_fwd(&mut self, size: usize, batch_size: usize, a: &Self) -> Result<(), Self::BaseError> {
        dense::pairwise(size, batch_size, a, self)
    }

    fn pairwise_bwd(&mut self, size: usize, batch_size: usize, a: &Self, grd: &Self) -> Result<(), Self::BaseError> {
        dense::backprop_pairwise(size, batch_size, a, grd, self)
    }

    fn power_error_fwd(&mut self, power: f32, size: usize, a: &Self, b: &Self) -> Result<(), Self::BaseError> {
        dense::abs_power_error(power, size, a, b, self)
    }

    fn power_error_bwd(
        &mut self,
        power: f32,
        size: usize,
        a: &Self,
        b: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError> {
        dense::backprop_abs_power_error_single(power, size, a, b, grd, self)
    }

    fn adam(
        &mut self,
        config: &AdamConfig,
        size: usize,
        grd: &Self,
        mom: &mut Self,
        vel: &mut Self,
    ) -> Result<(), Self::BaseError> {
        let AdamConfig { beta1, beta2, gradient_factor, learning_rate, denom } = *config;
        dense::adam(size, self, grd, mom, vel, beta1, beta2, gradient_factor, learning_rate, denom)
    }

    fn clip(&mut self, size: usize, min: f32, max: f32) -> Result<(), Self::BaseError> {
        dense::clip(size, self, min, max)
    }
}

impl Device for ExecutionContext {
    type BufferF32 = Buffer<f32>;
    type BufferI32 = Buffer<i32>;
    type DeviceError = DeviceError;
    type IdType = ();

    fn new(_: Self::IdType) -> Result<Self, DeviceError> {
        Ok(Self::default())
    }

    fn synchronise(&self) -> Result<(), DeviceError> {
        util::device_synchronise()
    }

    fn get_last_device_error(&self) -> Result<(), DeviceError> {
        util::get_last_error()
    }

    fn backprop_sparse_affine_activate(
        batch_size: usize,
        stride: Option<bool>,
        activation: Activation,
        input_a: &Self::BufferF32,
        input_a_grad: &mut Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        input_c_grad: Option<&mut Self::BufferF32>,
        input_c_batched: bool,
        outputs: &Self::BufferF32,
        output_grad: &Self::BufferF32,
    ) -> OperationResult {
        sparse::backprop_sparse_affine(
            batch_size,
            stride,
            activation,
            input_a,
            input_a_grad,
            shape_a,
            input_b,
            shape_b,
            nnz,
            input_c,
            input_c_grad,
            input_c_batched,
            outputs,
            output_grad,
        )
    }

    fn sparse_affine_activate(
        batch_size: usize,
        stride: Option<bool>,
        activation: Activation,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        input_c_batched: bool,
        output: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::sparse_affine(
            batch_size,
            stride,
            activation,
            input_a,
            shape_a,
            input_b,
            shape_b,
            nnz,
            input_c,
            input_c_batched,
            output,
        )
    }

    fn select(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::select(batch_size, input_size, output_size, input, indices, output)
    }

    fn select_backprop(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::select_backprop(batch_size, input_size, output_size, indices, output_grad, input_grad)
    }

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::sparse_to_dense(batch_size, size, nnz, sparse, dense)
    }

    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult {
        dense::softmax_across_batch(batch_size, single_size, input, output)
    }

    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult {
        dense::crossentropy(size, pred, target, output)
    }

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult {
        dense::backprop_softmax_crossentropy(size, softmaxed, target, output_grad, input_grad)
    }

    fn mask(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        inputs: &Self::BufferF32,
        masks: &Self::BufferI32,
        outputs: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::mask(batch_size, single_size, inputs, masks, nnz, outputs)
    }

    fn backprop_mask(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        output_grads: &Self::BufferF32,
        masks: &Self::BufferI32,
        input_grads: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::backprop_mask(batch_size, single_size, output_grads, masks, nnz, input_grads)
    }

    fn gather(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        inputs: &Self::BufferF32,
        indices: &Self::BufferI32,
        outputs: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::gather(batch_size, input_size, output_size, inputs, indices, outputs)
    }

    fn backprop_gather(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        output_grads: &Self::BufferF32,
        indices: &Self::BufferI32,
        input_grads: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::backprop_gather(batch_size, input_size, output_size, output_grads, indices, input_grads)
    }

    fn softmax_across_batch_masked(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        masks: &Self::BufferI32,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::softmax_across_batch_masked(batch_size, single_size, nnz, masks, input, output)
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
    ) -> OperationResult {
        sparse::crossentropy_masked(batch_size, single_size, nnz, masks, pred, target, output, error)
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
    ) -> OperationResult {
        sparse::backprop_softmax_crossentropy_masked(
            batch_size,
            single_size,
            nnz,
            masks,
            softmaxed,
            target,
            output_grad,
            input_grad,
        )
    }
}
