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

        if let Some(b) = b {
            for i in 0..m {
                y[base + i] = b[bias_stride * loc + i];
            }
        } else {
            for i in 0..m {
                y[base + i] = 0.0;
            }
        }

        for i in 0..nnz {
            let inp = x[nnz * loc + i];

            if inp == -1 {
                break;
            }

            for j in 0..m {
                y[base + j] += a[m * inp as usize + j];
            }
        }

        for i in 0..m {
            y[base + i] = op(y[base + i]);
        }
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

        for i in 0..m {
            grd[i] = op(y[base + i]) * yg[base + i];
        }

        if let Some(bg) = bg.as_mut() {
            for i in 0..m {
                bg[bias_stride * loc + i] += grd[i];
            }
        }

        for i in 0..nnz {
            let inp = x[nnz * loc + i];

            if inp == -1 {
                break;
            }

            for j in 0..m {
                ag[m * inp as usize + j] += grd[j];
            }
        }
    }
}
