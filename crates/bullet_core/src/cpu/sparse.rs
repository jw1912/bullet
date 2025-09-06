#![allow(clippy::too_many_arguments)]

use crate::{
    cpu::CpuThread,
    device::{DeviceBuffer, OperationError, OperationResult, SparseAffineOps},
    graph::{builder::Shape, ir::operation::unary::DiffableFromOutput},
};

impl SparseAffineOps for CpuThread {
    fn sparse_affine_activate(
        batch_size: usize,
        activation: DiffableFromOutput,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        input_b_vals: Option<&Self::BufferF32>,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        input_c_batched: bool,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let shape_o = shape_a * shape_b;

        if shape_a.size() > input_a.size()
            || batch_size * nnz > input_b.size()
            || batch_size * shape_o.size() > output.size()
        {
            return Err(OperationError::IndexOutOfBounds);
        }

        if let Some(c) = input_c {
            if shape_o.size() * if input_c_batched { batch_size } else { 1 } > c.size() {
                return Err(OperationError::IndexOutOfBounds);
            }
        }

        let m = shape_a.rows();
        let k = batch_size;
        let a = &input_a.buf;
        let x = &input_b.buf;
        let v = input_b_vals.map(|v| &*v.buf);
        let b = input_c.map(|c| &*c.buf);
        let bb = input_c_batched;
        let y = &mut output.buf;

        match activation {
            DiffableFromOutput::Identity => affine_fwd(nnz, m, k, a, x, v, b, bb, y, |x| x),
            DiffableFromOutput::ReLU => affine_fwd(nnz, m, k, a, x, v, b, bb, y, |x| x.max(0.0)),
            DiffableFromOutput::CReLU => affine_fwd(nnz, m, k, a, x, v, b, bb, y, |x| x.clamp(0.0, 1.0)),
            DiffableFromOutput::SCReLU => affine_fwd(nnz, m, k, a, x, v, b, bb, y, |x| x.clamp(0.0, 1.0).powi(2)),
            DiffableFromOutput::SqrReLU => affine_fwd(nnz, m, k, a, x, v, b, bb, y, |x| x.max(0.0).powi(2)),
            DiffableFromOutput::Sigmoid => affine_fwd(nnz, m, k, a, x, v, b, bb, y, |x| 1.0 / (1.0 + (-x).exp())),
        }

        Ok(())
    }

    fn backprop_sparse_affine_activate(
        batch_size: usize,
        activation: DiffableFromOutput,
        input_a_grad: &mut Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        input_b_vals: Option<&Self::BufferF32>,
        shape_b: Shape,
        nnz: usize,
        input_c_grad: Option<&mut Self::BufferF32>,
        input_c_batched: bool,
        outputs: &Self::BufferF32,
        output_grad: &Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let shape_o = shape_a * shape_b;

        assert_eq!(shape_b.cols(), 1);
        assert_eq!(shape_o.cols(), 1);
        if shape_a.size() > input_a_grad.size()
            || batch_size * nnz > input_b.size()
            || batch_size * shape_o.size() > outputs.size()
            || batch_size * shape_o.size() > output_grad.size()
        {
            return Err(OperationError::IndexOutOfBounds);
        }

        if let Some(ref grad) = input_c_grad {
            if shape_o.size() * if input_c_batched { batch_size } else { 1 } > grad.size() {
                return Err(OperationError::IndexOutOfBounds);
            }
        }

        let m = shape_a.rows();
        let k = batch_size;
        let x = &input_b.buf;
        let v = input_b_vals.map(|v| &*v.buf);
        let y = &outputs.buf;
        let yg = &output_grad.buf;
        let bb = input_c_batched;
        let ag = &mut input_a_grad.buf;
        let bg = input_c_grad.map(|x| &mut *x.buf);

        match activation {
            DiffableFromOutput::Identity => affine_bwd(nnz, m, k, x, v, y, yg, bb, ag, bg, |_| 1.0),
            DiffableFromOutput::ReLU => affine_bwd(nnz, m, k, x, v, y, yg, bb, ag, bg, |x| f32::from(x > 0.0)),
            DiffableFromOutput::CReLU => {
                affine_bwd(nnz, m, k, x, v, y, yg, bb, ag, bg, |x| if x > 0.0 && x < 1.0 { 1.0 } else { 0.0 })
            }
            DiffableFromOutput::SCReLU => {
                affine_bwd(
                    nnz,
                    m,
                    k,
                    x,
                    v,
                    y,
                    yg,
                    bb,
                    ag,
                    bg,
                    |x| {
                        if x > 0.0 && x < 1.0 { 2.0 * x.sqrt() } else { 0.0 }
                    },
                )
            }
            DiffableFromOutput::SqrReLU => affine_bwd(nnz, m, k, x, v, y, yg, bb, ag, bg, |x| 2.0 * x.max(0.0).sqrt()),
            DiffableFromOutput::Sigmoid => affine_bwd(nnz, m, k, x, v, y, yg, bb, ag, bg, |x| x * (1.0 - x)),
        }

        Ok(())
    }
}

pub fn affine_fwd<F: Fn(f32) -> f32>(
    nnz: usize,
    m: usize,
    k: usize,
    a: &[f32],
    x: &[i32],
    v: Option<&[f32]>,
    b: Option<&[f32]>,
    bb: bool,
    y: &mut [f32],
    op: F,
) {
    let bias_stride = if bb { m } else { 0 };

    for loc in 0..k {
        let base = m * loc;
        let ty = &mut y[base..base + m];
        let tx = &x[nnz * loc..nnz * loc + nnz];
        let tv = v.map(|v| &v[nnz * loc..nnz * loc + nnz]);
        let tb = b.map(|b| &b[bias_stride * loc..bias_stride * loc + m]);

        if m % 32 == 0 {
            affine_fwd_single_fast::<32, F>(m, a, tx, tv, tb, ty, &op);
        } else {
            affine_fwd_single_fallback(m, a, tx, tv, tb, ty, &op);
        }
    }
}

pub fn affine_bwd<F: Fn(f32) -> f32>(
    nnz: usize,
    m: usize,
    k: usize,
    x: &[i32],
    v: Option<&[f32]>,
    y: &[f32],
    yg: &[f32],
    bb: bool,
    ag: &mut [f32],
    mut bg: Option<&mut [f32]>,
    op: F,
) {
    let bias_stride = if bb { m } else { 0 };

    let mut grd = vec![0.0; m];

    for loc in 0..k {
        let base = m * loc;
        let tx = &x[nnz * loc..nnz * loc + nnz];
        let tv = v.map(|v| &v[nnz * loc..nnz * loc + nnz]);
        let ty = &y[base..base + m];
        let tyg = &yg[base..base + m];
        let tbg = bg.as_mut().map(|g| &mut g[bias_stride * loc..bias_stride * loc + m]);

        if m % 32 == 0 {
            affine_bwd_single_fast::<32, F>(m, tx, tv, ty, tyg, ag, tbg, &op);
        } else {
            affine_bwd_single_fallback(&mut grd, m, tx, tv, ty, tyg, ag, tbg, &op);
        }
    }
}

fn affine_fwd_single_fast<const T: usize, F: Fn(f32) -> f32>(
    m: usize,
    a: &[f32],
    x: &[i32],
    v: Option<&[f32]>,
    b: Option<&[f32]>,
    ty: &mut [f32],
    op: &F,
) {
    assert_eq!(m % T, 0);

    for p in 0..m / T {
        let d = T * p;
        let mut tt = [0.0; T];

        if let Some(b) = b {
            for (t, &tb) in tt.iter_mut().zip(b[d..d + T].iter()) {
                *t = tb;
            }
        }

        for (j, &inp) in x.iter().enumerate() {
            if inp == -1 {
                break;
            }

            let v = v.map(|v| v[j]).unwrap_or(1.0);
            let base = m * inp as usize + d;
            for (t, &ta) in tt.iter_mut().zip(a[base..base + T].iter()) {
                *t += v * ta;
            }
        }

        for (i, &j) in ty[d..d + T].iter_mut().zip(tt.iter()) {
            *i = op(j);
        }
    }
}

fn affine_fwd_single_fallback<F: Fn(f32) -> f32>(
    m: usize,
    a: &[f32],
    x: &[i32],
    v: Option<&[f32]>,
    b: Option<&[f32]>,
    ty: &mut [f32],
    op: &F,
) {
    if let Some(b) = b {
        by_chunks_32_2(ty, b, |_, b| b);
    } else {
        by_chunks_32_1(ty, |_| 0.0);
    }

    for (j, &inp) in x.iter().enumerate() {
        if inp == -1 {
            break;
        }

        let v = v.map(|v| v[j]).unwrap_or(1.0);
        let base = m * inp as usize;
        by_chunks_32_2(ty, &a[base..base + m], |a, b| a + v * b);
    }

    by_chunks_32_1(ty, op);
}

fn affine_bwd_single_fast<const T: usize, F: Fn(f32) -> f32>(
    m: usize,
    tx: &[i32],
    tv: Option<&[f32]>,
    ty: &[f32],
    tyg: &[f32],
    ag: &mut [f32],
    mut tbg: Option<&mut [f32]>,
    op: &F,
) {
    assert_eq!(m % T, 0);

    for p in 0..m / T {
        let b = p * T;

        let mut grd = [0.0; T];
        for ((i, &j), &k) in grd.iter_mut().zip(ty[b..b + T].iter()).zip(tyg[b..b + T].iter()) {
            *i = op(j) * k;
        }

        if let Some(tbg) = tbg.as_mut() {
            for (i, &j) in tbg[b..b + T].iter_mut().zip(grd.iter()) {
                *i += j;
            }
        }

        for (j, &inp) in tx.iter().enumerate() {
            if inp == -1 {
                break;
            }

            let v = tv.map(|v| v[j]).unwrap_or(1.0);
            let base = m * inp as usize + b;
            for (i, &j) in ag[base..base + T].iter_mut().zip(grd.iter()) {
                *i += v * j;
            }
        }
    }
}

fn affine_bwd_single_fallback<F: Fn(f32) -> f32>(
    grd: &mut [f32],
    m: usize,
    tx: &[i32],
    tv: Option<&[f32]>,
    ty: &[f32],
    tyg: &[f32],
    ag: &mut [f32],
    mut tbg: Option<&mut [f32]>,
    op: &F,
) {
    by_chunks_32_3(grd, ty, tyg, |_, a, b| op(a) * b);

    if let Some(tbg) = tbg.as_mut() {
        by_chunks_32_2(tbg, grd, |a, b| a + b);
    }

    for (j, &inp) in tx.iter().enumerate() {
        if inp == -1 {
            break;
        }

        let v = tv.map(|v| v[j]).unwrap_or(1.0);
        let base = m * inp as usize;
        by_chunks_32_2(&mut ag[base..base + m], grd, |a, b| a + v * b);
    }
}

pub fn by_chunks_32_1<F: Fn(f32) -> f32>(a: &mut [f32], f: F) {
    if a.len() % 32 == 0 {
        for ac in a.chunks_exact_mut(32) {
            for ai in ac {
                *ai = f(*ai);
            }
        }
    } else {
        for ac in a.chunks_mut(32) {
            for ai in ac {
                *ai = f(*ai);
            }
        }
    }
}

pub fn by_chunks_32_2<F: Fn(f32, f32) -> f32>(a: &mut [f32], b: &[f32], f: F) {
    assert_eq!(a.len(), b.len());

    if a.len() % 32 == 0 {
        for (ac, bc) in a.chunks_exact_mut(32).zip(b.chunks_exact(32)) {
            for (ai, &bi) in ac.iter_mut().zip(bc.iter()) {
                *ai = f(*ai, bi);
            }
        }
    } else {
        for (ac, bc) in a.chunks_mut(32).zip(b.chunks(32)) {
            for (ai, &bi) in ac.iter_mut().zip(bc.iter()) {
                *ai = f(*ai, bi);
            }
        }
    }
}

pub fn by_chunks_32_3<F: Fn(f32, f32, f32) -> f32>(a: &mut [f32], b: &[f32], c: &[f32], f: F) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());

    if a.len() % 32 == 0 {
        for ((ac, bc), cc) in a.chunks_exact_mut(32).zip(b.chunks_exact(32)).zip(c.chunks_exact(32)) {
            for ((ai, &bi), &ci) in ac.iter_mut().zip(bc.iter()).zip(cc.iter()) {
                *ai = f(*ai, bi, ci);
            }
        }
    } else {
        for ((ac, bc), cc) in a.chunks_mut(32).zip(b.chunks(32)).zip(c.chunks(32)) {
            for ((ai, &bi), &ci) in ac.iter_mut().zip(bc.iter()).zip(cc.iter()) {
                *ai = f(*ai, bi, ci);
            }
        }
    }
}
