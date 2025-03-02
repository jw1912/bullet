#![allow(unused)]
use std::sync::Arc;

use crate::{
    backend::{
        activation::Activation,
        error::{OperationError, OperationResult},
        shape::Shape,
        Device, DeviceBuffer,
    },
    graph::tests,
};

tests::make_tests! {
    CpuThread,
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

#[derive(Debug)]
pub struct CpuError;

pub struct CpuThread;

pub struct CpuBuffer<T> {
    buf: Vec<T>,
    device: Arc<CpuThread>,
}

impl<T: Copy + Default> DeviceBuffer<CpuThread, T> for CpuBuffer<T> {
    fn device(&self) -> Arc<CpuThread> {
        self.device.clone()
    }

    fn new(device: Arc<CpuThread>, size: usize) -> Result<Self, CpuError> {
        Ok(Self { buf: vec![T::default(); size], device })
    }

    fn size(&self) -> usize {
        self.buf.len()
    }

    fn set_zero(&mut self) -> Result<(), CpuError> {
        for elem in &mut self.buf {
            *elem = T::default();
        }

        Ok(())
    }

    fn load_from_device(&mut self, buf: &Self, num: usize) -> Result<(), CpuError> {
        self.buf[..num].copy_from_slice(&buf.buf[..num]);
        Ok(())
    }

    fn load_from_slice(&mut self, buf: &[T]) -> Result<(), CpuError> {
        self.buf[..buf.len()].copy_from_slice(buf);
        Ok(())
    }

    fn write_into_slice(&self, buf: &mut [T], num: usize) -> Result<(), CpuError> {
        buf[..num].copy_from_slice(&self.buf[..num]);
        Ok(())
    }
}

impl Device for CpuThread {
    type BufferF32 = CpuBuffer<f32>;
    type BufferI32 = CpuBuffer<i32>;

    type DeviceError = CpuError;

    type IdType = ();

    fn new(_id: Self::IdType) -> Result<Self, Self::DeviceError> {
        Ok(Self)
    }

    fn synchronise(&self) -> Result<(), Self::DeviceError> {
        Ok(())
    }

    fn get_last_device_error(&self) -> Result<(), Self::DeviceError> {
        Ok(())
    }

    fn activate(
        size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
        activation: Activation,
    ) -> OperationResult<Self::DeviceError> {
        fn apply<F: Fn(f32) -> f32>(size: usize, input: &CpuBuffer<f32>, output: &mut CpuBuffer<f32>, f: F) {
            for (o, &i) in output.buf[..size].iter_mut().zip(input.buf[..size].iter()) {
                *o = f(i);
            }
        }

        match activation {
            Activation::Identity => apply(size, input, output, |x| x),
            Activation::ReLU => apply(size, input, output, |x| x.max(0.0)),
            Activation::CReLU => apply(size, input, output, |x| x.clamp(0.0, 1.0)),
            Activation::SCReLU => apply(size, input, output, |x| x.clamp(0.0, 1.0).powi(2)),
            Activation::SqrReLU => apply(size, input, output, |x| x.max(0.0).powi(2)),
            Activation::Square => apply(size, input, output, |x| x.powi(2)),
            Activation::Sigmoid => apply(size, input, output, |x| 1.0 / (1.0 + (-x).exp())),
        }

        Ok(())
    }

    fn backprop_activate(
        size: usize,
        input: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
        output_grad: &Self::BufferF32,
        activation: Activation,
    ) -> OperationResult<Self::DeviceError> {
        fn apply<F: Fn(f32) -> f32>(
            size: usize,
            input: &CpuBuffer<f32>,
            output_grad: &CpuBuffer<f32>,
            input_grad: &mut CpuBuffer<f32>,
            f: F,
        ) {
            for ((ig, &og), &i) in
                input_grad.buf[..size].iter_mut().zip(output_grad.buf[..size].iter()).zip(input.buf[..size].iter())
            {
                *ig += f(i) * og;
            }
        }

        match activation {
            Activation::Identity => apply(size, input, output_grad, input_grad, |x| 1.0),
            Activation::ReLU => apply(size, input, output_grad, input_grad, |x| f32::from(x > 0.0)),
            Activation::CReLU => apply(size, input, output_grad, input_grad, |x| f32::from(x > 0.0 && x < 1.0)),
            Activation::SCReLU => {
                apply(size, input, output_grad, input_grad, |x| if x > 0.0 && x < 1.0 { 2.0 * x } else { 0.0 })
            }
            Activation::SqrReLU => apply(size, input, output_grad, input_grad, |x| if x > 0.0 { 2.0 * x } else { 0.0 }),
            Activation::Square => apply(size, input, output_grad, input_grad, |x| 2.0 * x),
            Activation::Sigmoid => apply(size, input, output_grad, input_grad, |x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                sig * (1.0 - sig)
            }),
        }

        Ok(())
    }

    fn sgemm(
        alpha: f32,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        trans_a: bool,
        input_b: &Self::BufferF32,
        shape_b: Shape,
        trans_b: bool,
        beta: f32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
        sgemm(
            alpha,
            &input_a.buf[..shape_a.size()],
            shape_a,
            trans_a,
            &input_b.buf[..shape_b.size()],
            shape_b,
            trans_b,
            beta,
            &mut output.buf[..shape_o.size()],
        )
    }

    fn sgemm_batched(
        batch_size: usize,
        alpha: f32,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        trans_a: bool,
        input_b: &Self::BufferF32,
        shape_b: Shape,
        trans_b: bool,
        beta: f32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

        for ((o, ia), ib) in output.buf[..shape_o.size() * batch_size]
            .chunks_exact_mut(shape_o.size())
            .zip(input_a.buf[..shape_a.size() * batch_size].chunks_exact(shape_o.size()))
            .zip(input_b.buf[..shape_b.size() * batch_size].chunks_exact(shape_o.size()))
        {
            sgemm(alpha, ia, shape_a, trans_a, ib, shape_b, trans_b, beta, o)?;
        }

        Ok(())
    }

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
    ) -> OperationResult<Self::DeviceError> {
        match (input_a, input_b) {
            (Some(a), Some(b)) => {
                for ((o, &a), &b) in output.buf[..size].iter_mut().zip(a.buf[..size].iter()).zip(b.buf[..size].iter()) {
                    *o = alpha * a + beta * b;
                }
            }
            (Some(a), None) => {
                for (o, &a) in output.buf[..size].iter_mut().zip(a.buf[..size].iter()) {
                    *o = alpha * a;
                }
            }
            (None, Some(b)) => {
                for (o, &b) in output.buf[..size].iter_mut().zip(b.buf[..size].iter()) {
                    *o = alpha * *o + beta * b;
                }
            }
            (None, None) => {
                for o in &mut output.buf[..size] {
                    *o *= alpha;
                }
            }
        }

        Ok(())
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
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
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
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
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
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn mask(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        inputs: &Self::BufferF32,
        masks: &Self::BufferI32,
        outputs: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn backprop_mask(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        output_grads: &Self::BufferF32,
        masks: &Self::BufferI32,
        input_grads: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn pairwise(
        single_size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
        post_concat: bool,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn backprop_pairwise(
        single_size: usize,
        batch_size: usize,
        input: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
        post_concat: bool,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn select(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn select_backprop(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn gather(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        inputs: &Self::BufferF32,
        indices: &Self::BufferI32,
        outputs: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn backprop_gather(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        output_grads: &Self::BufferF32,
        indices: &Self::BufferI32,
        input_grads: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn abs_power_error(
        power: f32,
        size: usize,
        input_a: &Self::BufferF32,
        input_b: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        for ((o, &a), &b) in
            output.buf[..size].iter_mut().zip(input_a.buf[..size].iter()).zip(input_b.buf[..size].iter())
        {
            *o = (a - b).abs().powf(power)
        }

        Ok(())
    }

    fn backprop_abs_power_error_single(
        power: f32,
        size: usize,
        input_a: &Self::BufferF32,
        input_b: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_a_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        for (((ig, &og), &ia), &ib) in input_a_grad.buf[..size]
            .iter_mut()
            .zip(output_grad.buf[..size].iter())
            .zip(input_a.buf[..size].iter())
            .zip(input_b.buf[..size].iter())
        {
            let diff = ia - ib;
            let grad = power * diff.abs().powf(power - 1.0) * og;
            *ig += grad * diff.signum();
        }

        Ok(())
    }

    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    fn softmax_across_batch_masked(
        batch_size: usize,
        single_size: usize,
        nnz: usize,
        masks: &Self::BufferI32,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
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
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
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
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

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
    ) -> OperationResult<Self::DeviceError> {
        for (((p, &g), m), v) in params.buf[..size]
            .iter_mut()
            .zip(gradient.buf[..size].iter())
            .zip(momentum.buf[..size].iter_mut())
            .zip(velocity.buf[..size].iter_mut())
        {
            let grad = gradient_factor * g;
            *m = beta1 * *m + (1.0 - beta1) * grad;
            *v = beta2 * *v + (1.0 - beta2) * grad * grad;

            let mut val = *m;
            if denom {
                val /= (*v + 0.00000001).sqrt();
            }

            *p -= learning_rate * val;
        }

        Ok(())
    }

    fn clip(size: usize, params: &mut Self::BufferF32, min: f32, max: f32) -> OperationResult<Self::DeviceError> {
        for p in &mut params.buf[..size] {
            *p = p.clamp(min, max);
        }

        Ok(())
    }

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }
}

#[allow(clippy::too_many_arguments)]
fn sgemm(
    alpha: f32,
    input_a: &[f32],
    shape_a: Shape,
    trans_a: bool,
    input_b: &[f32],
    shape_b: Shape,
    trans_b: bool,
    beta: f32,
    output: &mut [f32],
) -> OperationResult<CpuError> {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    if input_a.len() != shape_a.size() || input_b.len() != shape_b.size() || output.len() != shape_o.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    match (trans_a, trans_b) {
        (false, false) => {
            mm::<false, false>(shape_a.rows(), shape_a.cols(), shape_b.cols(), alpha, input_a, input_b, beta, output)
        }
        (false, true) => {
            mm::<false, true>(shape_a.rows(), shape_a.cols(), shape_b.rows(), alpha, input_a, input_b, beta, output)
        }
        (true, false) => {
            mm::<true, false>(shape_a.cols(), shape_a.rows(), shape_b.cols(), alpha, input_a, input_b, beta, output)
        }
        (true, true) => {
            mm::<true, true>(shape_a.cols(), shape_a.rows(), shape_b.rows(), alpha, input_a, input_b, beta, output)
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mm<const TA: bool, const TB: bool>(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    input_a: &[f32],
    input_b: &[f32],
    beta: f32,
    output: &mut [f32],
) {
    for ki in 0..k {
        for mi in 0..m {
            let mut sum = 0.0;
            for ni in 0..n {
                let aidx = if TA { n * mi + ni } else { m * ni + mi };
                let bidx = if TB { k * ni + ki } else { n * ki + ni };
                sum += input_a[aidx] * input_b[bidx];
            }
            output[m * ki + mi] = alpha * sum + beta * output[m * ki + mi];
        }
    }
}
