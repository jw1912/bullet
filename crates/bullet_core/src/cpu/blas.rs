#![allow(clippy::too_many_arguments)]
use crate::{
    cpu::sparse::by_chunks_32_2,
    device::blas::{BlasOperations, GemmConfig},
    graph::ir::shape::Shape,
};

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
}

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

    mm_internal::<TA, TB>(m, n, k, alpha, a, b, beta, c);

    Ok(())
}

fn mm_internal<const TA: bool, const TB: bool>(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    // forward optimisation
    if !TA && !TB && alpha == 1.0 && beta == 0.0 && m == 1 {
        return mm_nn_m1(n, a, b, c);
    }

    // backprop optmisations
    if alpha == 1.0 && beta == 1.0 {
        if !TA && TB && m == 1 {
            return mm_nt_m1(k, a, b, c);
        } else if TA && !TB && n == 1 {
            return mm_tn_n1(m, a, b, c);
        }
    }

    mm_fallback::<TA, TB>(m, n, k, alpha, a, b, beta, c);
}

fn mm_fallback<const TA: bool, const TB: bool>(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
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
}

fn mm_nn_m1(n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for (loc, tc) in c.iter_mut().enumerate() {
        let mut sum = [0.0; 32];

        by_chunks_32_2_mut(a, &b[loc * n..(loc + 1) * n], |i, ta, tb| sum[i] += ta * tb);

        *tc = sum.iter().sum();
    }
}

fn mm_nt_m1(k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for (loc, &ta) in a.iter().enumerate() {
        by_chunks_32_2(c, &b[loc * k..(loc + 1) * k], |c, tb| c + ta * tb);
    }
}

fn mm_tn_n1(m: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for (loc, &tb) in b.iter().enumerate() {
        by_chunks_32_2(&mut c[loc * m..(loc + 1) * m], a, |c, ta| c + ta * tb);
    }
}

fn by_chunks_32_2_mut<F: FnMut(usize, f32, f32)>(a: &[f32], b: &[f32], mut f: F) {
    assert_eq!(a.len(), b.len());

    if a.len() % 32 == 0 {
        for (ac, bc) in a.chunks_exact(32).zip(b.chunks_exact(32)) {
            for (i, (&ai, &bi)) in ac.iter().zip(bc.iter()).enumerate() {
                f(i, ai, bi);
            }
        }
    } else {
        for (ac, bc) in a.chunks(32).zip(b.chunks(32)) {
            for (i, (&ai, &bi)) in ac.iter().zip(bc.iter()).enumerate() {
                f(i, ai, bi);
            }
        }
    }
}
