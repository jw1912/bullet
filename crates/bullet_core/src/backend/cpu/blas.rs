use crate::backend::device::blas::{BlasOperations, GemmConfig, Shape};

use super::{CpuBuffer, CpuError};

impl BlasOperations for CpuBuffer<f32> {
    type BlasError = CpuError;

    fn gemm(&mut self, config: &GemmConfig, a: &Self, b: &Self) -> Result<(), Self::BlasError> {
        let GemmConfig { alpha, beta, shape_a, trans_a, shape_b, trans_b } = *config;
        let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
        sgemm(
            alpha,
            &a.buf[..shape_a.size()],
            shape_a,
            trans_a,
            &b.buf[..shape_b.size()],
            shape_b,
            trans_b,
            beta,
            &mut self.buf[..shape_o.size()],
        )
    }

    fn gebmm(&mut self, config: &GemmConfig, batch_size: usize, a: &Self, b: &Self) -> Result<(), Self::BlasError> {
        let GemmConfig { alpha, beta, shape_a, trans_a, shape_b, trans_b } = *config;

        let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

        for ((o, ia), ib) in self.buf[..shape_o.size() * batch_size]
            .chunks_exact_mut(shape_o.size())
            .zip(a.buf[..shape_a.size() * batch_size].chunks_exact(shape_a.size()))
            .zip(b.buf[..shape_b.size() * batch_size].chunks_exact(shape_b.size()))
        {
            sgemm(alpha, ia, shape_a, trans_a, ib, shape_b, trans_b, beta, o)?;
        }

        Ok(())
    }

    fn geam(
        &mut self,
        size: usize,
        alpha: f32,
        a: Option<&Self>,
        beta: f32,
        b: Option<&Self>,
    ) -> Result<(), Self::BlasError> {
        match (a, b) {
            (Some(a), Some(b)) => {
                for ((o, &a), &b) in self.buf[..size].iter_mut().zip(a.buf[..size].iter()).zip(b.buf[..size].iter()) {
                    *o = alpha * a + beta * b;
                }
            }
            (Some(a), None) => {
                for (o, &a) in self.buf[..size].iter_mut().zip(a.buf[..size].iter()) {
                    *o = alpha * a;
                }
            }
            (None, Some(b)) => {
                for (o, &b) in self.buf[..size].iter_mut().zip(b.buf[..size].iter()) {
                    *o = alpha * *o + beta * b;
                }
            }
            (None, None) => {
                for o in &mut self.buf[..size] {
                    *o *= alpha;
                }
            }
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn sgemm(
    alpha: f32,
    a: &[f32],
    sha: Shape,
    ta: bool,
    b: &[f32],
    shb: Shape,
    tb: bool,
    beta: f32,
    c: &mut [f32],
) -> Result<(), CpuError> {
    match (ta, tb) {
        (false, false) => mm::<false, false>(sha.rows(), sha.cols(), shb.cols(), alpha, a, b, beta, c),
        (false, true) => mm::<false, true>(sha.rows(), sha.cols(), shb.rows(), alpha, a, b, beta, c),
        (true, false) => mm::<true, false>(sha.cols(), sha.rows(), shb.cols(), alpha, a, b, beta, c),
        (true, true) => mm::<true, true>(sha.cols(), sha.rows(), shb.rows(), alpha, a, b, beta, c),
    }
}

#[allow(clippy::too_many_arguments)]
fn mm<const TA: bool, const TB: bool>(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<(), CpuError> {
    if a.len() != m * n || b.len() != n * k || c.len() != m * k {
        return Err(CpuError);
    }

    for ki in 0..k {
        for mi in 0..m {
            let mut sum = 0.0;
            for ni in 0..n {
                let aidx = if TA { n * mi + ni } else { m * ni + mi };
                let bidx = if TB { k * ni + ki } else { n * ki + ni };
                sum += a[aidx] * b[bidx];
            }
            c[m * ki + mi] = alpha * sum + beta * c[m * ki + mi];
        }
    }

    Ok(())
}
