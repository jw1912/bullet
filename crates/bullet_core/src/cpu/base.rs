use crate::{
    device::base::{AdamConfig, BaseOperations},
    graph::ir::operation::unary::DiffableFromOutput,
};

use super::{CpuBuffer, CpuError};

impl BaseOperations for CpuBuffer<f32> {
    type BaseError = CpuError;

    fn set_to(&mut self, size: usize, val: f32) -> Result<(), Self::BaseError> {
        if size > self.buf.len() {
            return Err(CpuError);
        }

        for i in self.buf.iter_mut().take(size) {
            *i = val;
        }

        Ok(())
    }

    fn diffable_from_output_fwd(
        &mut self,
        size: usize,
        a: &Self,
        act: DiffableFromOutput,
    ) -> Result<(), Self::BaseError> {
        fn apply<F: Fn(f32) -> f32>(size: usize, input: &CpuBuffer<f32>, output: &mut CpuBuffer<f32>, f: F) {
            for (o, &i) in output.buf[..size].iter_mut().zip(input.buf[..size].iter()) {
                *o = f(i);
            }
        }

        match act {
            DiffableFromOutput::Identity => apply(size, a, self, |x| x),
            DiffableFromOutput::ReLU => apply(size, a, self, |x| x.max(0.0)),
            DiffableFromOutput::CReLU => apply(size, a, self, |x| x.clamp(0.0, 1.0)),
            DiffableFromOutput::SCReLU => apply(size, a, self, |x| x.clamp(0.0, 1.0).powi(2)),
            DiffableFromOutput::SqrReLU => apply(size, a, self, |x| x.max(0.0).powi(2)),
            DiffableFromOutput::Sigmoid => apply(size, a, self, |x| 1.0 / (1.0 + (-x).exp())),
        }

        Ok(())
    }

    fn diffable_from_output_bwd(
        &mut self,
        size: usize,
        a: &Self,
        grd: &Self,
        act: DiffableFromOutput,
    ) -> Result<(), Self::BaseError> {
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

        match act {
            DiffableFromOutput::Identity => apply(size, a, grd, self, |_| 1.0),
            DiffableFromOutput::ReLU => apply(size, a, grd, self, |x| f32::from(x > 0.0)),
            DiffableFromOutput::CReLU => apply(size, a, grd, self, |x| f32::from(x > 0.0 && x < 1.0)),
            DiffableFromOutput::SCReLU => apply(size, a, grd, self, |x| if x > 0.0 && x < 1.0 { 2.0 * x } else { 0.0 }),
            DiffableFromOutput::SqrReLU => apply(size, a, grd, self, |x| if x > 0.0 { 2.0 * x } else { 0.0 }),
            DiffableFromOutput::Sigmoid => apply(size, a, grd, self, |x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                sig * (1.0 - sig)
            }),
        }

        Ok(())
    }

    fn linear_comb(&mut self, size: usize, alpha: f32, beta: f32, nb: &Self) -> Result<(), Self::BaseError> {
        for (o, &i) in self.buf[..size].iter_mut().zip(nb.buf[..size].iter()) {
            *o = alpha * *o + beta * i;
        }

        Ok(())
    }

    fn linear_comb_splat(
        &mut self,
        size: usize,
        batch_size: usize,
        alpha: f32,
        beta: f32,
        nb: &Self,
    ) -> Result<(), Self::BaseError> {
        for single in self.buf.chunks_exact_mut(size).take(batch_size) {
            for (o, &i) in single.iter_mut().zip(nb.buf[..size].iter()) {
                *o = alpha * *o + beta * i;
            }
        }

        Ok(())
    }

    fn reduce_across_batch(
        &mut self,
        size: usize,
        batch_size: usize,
        output_mul: f32,
        input_mul: f32,
        input: &Self,
    ) -> Result<(), Self::BaseError> {
        for o in &mut self.buf[..size] {
            *o *= output_mul;
        }

        for single in input.buf.chunks_exact(size).take(batch_size) {
            for (o, &i) in self.buf[..size].iter_mut().zip(single.iter()) {
                *o += input_mul * i;
            }
        }

        Ok(())
    }

    fn mul_scalar(&mut self, size: usize, alpha: f32) -> Result<(), Self::BaseError> {
        for x in &mut self.buf[..size] {
            *x *= alpha;
        }

        Ok(())
    }

    fn add_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError> {
        for (o, &i) in self.buf[..size].iter_mut().zip(input.buf[..size].iter()) {
            *o = i + alpha;
        }

        Ok(())
    }

    fn abs_pow_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError> {
        for (o, &i) in self.buf[..size].iter_mut().zip(input.buf[..size].iter()) {
            *o = i.abs().powf(alpha);
        }

        Ok(())
    }

    fn abs_pow_scalar_backward(
        &mut self,
        size: usize,
        alpha: f32,
        input: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError> {
        for ((ig, &og), &i) in self.buf[..size].iter_mut().zip(grd.buf[..size].iter()).zip(input.buf[..size].iter()) {
            let err = alpha * i.abs().powf(alpha - 1.0);
            *ig += og * if i > 0.0 { err } else { -err };
        }

        Ok(())
    }

    fn pairwise_fwd(&mut self, size: usize, batch_size: usize, a: &Self) -> Result<(), Self::BaseError> {
        for i in 0..batch_size {
            for j in 0..size / 2 {
                let k = i * size + j;
                self.buf[i * size / 2 + j] = a.buf[k] * a.buf[k + size / 2];
            }
        }

        Ok(())
    }

    fn pairwise_bwd(&mut self, size: usize, batch_size: usize, a: &Self, grd: &Self) -> Result<(), Self::BaseError> {
        for i in 0..batch_size {
            for j in 0..size / 2 {
                let g = grd.buf[i * size / 2 + j];
                let k = i * size + j;
                self.buf[k] += g * a.buf[k + size / 2];
                self.buf[k + size / 2] += g * a.buf[k];
            }
        }

        Ok(())
    }

    fn power_error_fwd(&mut self, power: f32, size: usize, a: &Self, b: &Self) -> Result<(), Self::BaseError> {
        for ((o, &a), &b) in self.buf[..size].iter_mut().zip(a.buf[..size].iter()).zip(b.buf[..size].iter()) {
            *o = (a - b).abs().powf(power)
        }

        Ok(())
    }

    fn power_error_bwd(
        &mut self,
        power: f32,
        size: usize,
        a: &Self,
        b: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError> {
        for (((ig, &og), &ia), &ib) in
            self.buf[..size].iter_mut().zip(grd.buf[..size].iter()).zip(a.buf[..size].iter()).zip(b.buf[..size].iter())
        {
            let diff = ia - ib;
            let grad = power * diff.abs().powf(power - 1.0) * og;
            *ig += grad * diff.signum();
        }

        Ok(())
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
        #[allow(clippy::too_many_arguments)]
        fn internal<const ADD: bool>(
            out: &mut CpuBuffer<f32>,
            rows: usize,
            cols: usize,
            offset: usize,
            stride: usize,
            a: &CpuBuffer<f32>,
            offset_a: usize,
            stride_a: usize,
        ) {
            for c in 0..cols {
                let oidx = offset + stride * c;
                let aidx = offset_a + stride_a * c;
                for r in 0..rows {
                    if ADD {
                        out.buf[oidx + r] += a.buf[aidx + r];
                    } else {
                        out.buf[oidx + r] = a.buf[aidx + r];
                    }
                }
            }
        }

        if add {
            internal::<true>(self, rows, cols, offset, stride, a, offset_a, stride_a);
        } else {
            internal::<false>(self, rows, cols, offset, stride, a, offset_a, stride_a);
        }

        Ok(())
    }

    fn clip(&mut self, size: usize, min: f32, max: f32) -> Result<(), Self::BaseError> {
        for p in &mut self.buf[..size] {
            *p = p.clamp(min, max);
        }

        Ok(())
    }

    fn adam(
        &mut self,
        config: &AdamConfig,
        size: usize,
        grd: &Self,
        mom: &mut Self,
        vel: &mut Self,
    ) -> Result<(), Self::BaseError> {
        let AdamConfig { beta1, beta2, gradient_factor, learning_rate, denom, decay, clip } = *config;
        for (((p, &g), m), v) in self.buf[..size]
            .iter_mut()
            .zip(grd.buf[..size].iter())
            .zip(mom.buf[..size].iter_mut())
            .zip(vel.buf[..size].iter_mut())
        {
            *p *= decay;

            let grad = gradient_factor * g;
            *m = beta1 * *m + (1.0 - beta1) * grad;
            *v = beta2 * *v + (1.0 - beta2) * grad * grad;

            let mut val = *m;
            if denom {
                val /= v.sqrt() + 0.00000001;
            }

            *p -= learning_rate * val;

            if let Some((min, max)) = clip {
                *p = (*p).clamp(min, max);
            }
        }

        Ok(())
    }
}
