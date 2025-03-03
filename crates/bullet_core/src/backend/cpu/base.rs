use crate::backend::device::base::{Activation, AdamConfig, BaseOperations};

use super::{CpuBuffer, CpuError};

impl BaseOperations for CpuBuffer<f32> {
    type BaseError = CpuError;

    fn activate_fwd(&mut self, size: usize, a: &Self, act: Activation) -> Result<(), Self::BaseError> {
        fn apply<F: Fn(f32) -> f32>(size: usize, input: &CpuBuffer<f32>, output: &mut CpuBuffer<f32>, f: F) {
            for (o, &i) in output.buf[..size].iter_mut().zip(input.buf[..size].iter()) {
                *o = f(i);
            }
        }

        match act {
            Activation::Identity => apply(size, a, self, |x| x),
            Activation::ReLU => apply(size, a, self, |x| x.max(0.0)),
            Activation::CReLU => apply(size, a, self, |x| x.clamp(0.0, 1.0)),
            Activation::SCReLU => apply(size, a, self, |x| x.clamp(0.0, 1.0).powi(2)),
            Activation::SqrReLU => apply(size, a, self, |x| x.max(0.0).powi(2)),
            Activation::Square => apply(size, a, self, |x| x.powi(2)),
            Activation::Sigmoid => apply(size, a, self, |x| 1.0 / (1.0 + (-x).exp())),
        }

        Ok(())
    }

    fn activate_bwd(&mut self, size: usize, a: &Self, grd: &Self, act: Activation) -> Result<(), Self::BaseError> {
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
            Activation::Identity => apply(size, a, grd, self, |x| 1.0),
            Activation::ReLU => apply(size, a, grd, self, |x| f32::from(x > 0.0)),
            Activation::CReLU => apply(size, a, grd, self, |x| f32::from(x > 0.0 && x < 1.0)),
            Activation::SCReLU => apply(size, a, grd, self, |x| if x > 0.0 && x < 1.0 { 2.0 * x } else { 0.0 }),
            Activation::SqrReLU => apply(size, a, grd, self, |x| if x > 0.0 { 2.0 * x } else { 0.0 }),
            Activation::Square => apply(size, a, grd, self, |x| 2.0 * x),
            Activation::Sigmoid => apply(size, a, grd, self, |x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                sig * (1.0 - sig)
            }),
        }

        Ok(())
    }

    fn pairwise_fwd(&mut self, size: usize, batch_size: usize, a: &Self) -> Result<(), Self::BaseError> {
        for (o, a) in self.buf.chunks_exact_mut(size / 2).zip(a.buf.chunks_exact(size)) {
            for i in 0..size / 2 {
                o[i] = a[i] * a[i + size / 2];
            }
        }

        Ok(())
    }

    fn pairwise_bwd(&mut self, size: usize, batch_size: usize, a: &Self, grd: &Self) -> Result<(), Self::BaseError> {
        for ((o, a), g) in
            self.buf.chunks_exact_mut(size).zip(a.buf.chunks_exact(size)).zip(grd.buf.chunks_exact(size / 2))
        {
            for i in 0..size / 2 {
                o[i] = g[i] * a[i + size / 2];
                o[i + size / 2] = g[i] * a[i]
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
            add: bool,
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
            internal::<true>(self, add, rows, cols, offset, stride, a, offset_a, stride_a);
        } else {
            internal::<false>(self, add, rows, cols, offset, stride, a, offset_a, stride_a);
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
        let AdamConfig { beta1, beta2, gradient_factor, learning_rate, denom } = *config;
        for (((p, &g), m), v) in self.buf[..size]
            .iter_mut()
            .zip(grd.buf[..size].iter())
            .zip(mom.buf[..size].iter_mut())
            .zip(vel.buf[..size].iter_mut())
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
}
