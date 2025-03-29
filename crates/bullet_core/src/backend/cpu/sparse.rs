#[allow(clippy::too_many_arguments)]
pub fn affine_fwd<F: Fn(f32) -> f32>(
    stride: usize,
    offset: usize,
    nnz: usize,
    m: usize,
    k: usize,
    a: &[f32],
    x: &[i32],
    b: Option<&[f32]>,
    bb: bool,
    y: &mut [f32],
    op: F,
) {
    let bias_stride = if bb { m } else { 0 };

    for loc in 0..k {
        let base = stride * m * loc + offset;
        let ty = &mut y[base..base + m];

        if let Some(b) = b {
            let base = bias_stride * loc;
            by_chunks_32_2(ty, &b[base..base + m], |_, b| b);
        } else {
            by_chunks_32_1(ty, |_| 0.0);
        }

        for i in 0..nnz {
            let inp = x[nnz * loc + i];

            if inp == -1 {
                break;
            }

            let base = m * inp as usize;
            by_chunks_32_2(ty, &a[base..base + m], |a, b| a + b);
        }

        by_chunks_32_1(ty, &op);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn affine_bwd<F: Fn(f32) -> f32>(
    stride: usize,
    offset: usize,
    nnz: usize,
    m: usize,
    k: usize,
    x: &[i32],
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
        let base = stride * m * loc + offset;

        by_chunks_32_3(&mut grd, &y[base..base + m], &yg[base..base + m], |_, a, b| op(a) * b);

        if let Some(bg) = bg.as_mut() {
            let base = bias_stride * loc;
            by_chunks_32_2(&mut bg[base..base + m], &grd, |a, b| a + b);
        }

        for i in 0..nnz {
            let inp = x[nnz * loc + i];

            if inp == -1 {
                break;
            }

            let base = m * inp as usize;
            by_chunks_32_2(&mut ag[base..base + m], &grd, |a, b| a + b);
        }
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
