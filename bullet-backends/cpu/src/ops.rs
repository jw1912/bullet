#![allow(unused_variables, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod bufops;
mod backprops;
mod mse;
mod sparse_affine;
mod splat_add;
mod update;

use crate::DeviceHandles;

pub use bufops::*;
pub use backprops::*;
pub use mse::*;
pub use sparse_affine::*;
pub use splat_add::*;
pub use update::*;

#[cfg(not(feature = "blas"))]
use crate::util;

#[cfg(feature = "blas")]
use bullet_blas::{cblas_sgemm, blasint, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

pub unsafe fn splat_mul_matrix_vector(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    batch_size: usize,
) {
    let a_ptr = a_ptr as usize;
    let x_ptr = x_ptr as usize;
    let y_ptr = y_ptr as usize;

    handle.split_workload(batch_size, |_, idx| {
        let a_ptr = a_ptr as *const f32;
        let x_ptr = (x_ptr as *const f32).add(m * idx);
        let y_ptr = (y_ptr as *mut f32).add(n * idx);

        for j in 0..n {
            let mut y = 0.0;
            let a = a_ptr.add(j);
            for i in 0..m {
                y += *a.add(n * i) * *x_ptr.add(i);
            }

            *y_ptr.add(j) = y;
        }
    });
}

pub unsafe fn splat_mul_matrixt_vector(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    y_ptr: *const f32,
    x_ptr: *mut f32,
    batch_size: usize,
) {
    let a_ptr = a_ptr as usize;
    let x_ptr = x_ptr as usize;
    let y_ptr = y_ptr as usize;

    handle.split_workload(batch_size, |_, idx| {
        let a_ptr = a_ptr as *const f32;
        let x_ptr = (x_ptr as *mut f32).add(m * idx);
        let y_ptr = (y_ptr as *const f32).add(n * idx);

        for i in 0..m {
            let mut x = 0.0;
            let col = a_ptr.add(i * n);

            for j in 0..n {
                x += *col.add(j) * *y_ptr.add(j);
            }

            *x_ptr.add(i) = x;
        }
    });
}

#[cfg(feature = "blas")]
pub unsafe fn reduce_add_mul_vector_vectort(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    y_ptr: *const f32,
    x_ptr: *const f32,
    a_ptr: *mut f32,
    batch_size: usize,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let m = m as blasint;
    let n = n as blasint;
    let batch_size = batch_size as blasint;

    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasConjNoTrans,
            CBLAS_TRANSPOSE::CblasConjTrans,
            n,
            m,
            batch_size,
            alpha,
            y_ptr,
            n,
            x_ptr,
            m,
            beta,
            a_ptr,
            n,
        );
    }
}

#[cfg(not(feature = "blas"))]
pub unsafe fn reduce_add_mul_vector_vectort(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    y_ptr: *const f32,
    x_ptr: *const f32,
    a_ptr: *mut f32,
    batch_size: usize,
) {
    let size = m * n;
    let x_ptr = x_ptr as usize;
    let y_ptr = y_ptr as usize;

    let mut a_ptrs = vec![0; handle.threads];
    for a in a_ptrs.iter_mut() {
        *a = util::calloc::<f32>(size) as usize;
    }

    handle.split_workload(batch_size, |thread, idx| {
        let a = a_ptrs[thread] as *mut f32;
        let x = (x_ptr as *const f32).add(idx * m);
        let y = (y_ptr as *const f32).add(idx * n);

        for i in 0..m {
            let col = a.add(i * n);
            let xi = *x.add(i);
            for j in 0..n {
                *col.add(j) += xi * *y.add(j);
            }
        }
    });

    for &a in a_ptrs.iter() {
        for i in 0..size {
            *a_ptr.add(i) += *(a as *const f32).add(i);
        }
    }

    for &a in a_ptrs.iter() {
        unsafe {
            util::free(a as *mut f32, size);
        }
    }
}

pub unsafe fn reduce_add(
    handle: DeviceHandles,
    _: *const f32,
    batch_size: usize,
    out_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    let inp = inp as usize;
    let out = out as usize;

    handle.split_workload(out_size, |_, idx| {
        let this_inp = (inp as *const f32).add(idx);

        let mut sum = 0.0;

        for i in 0..batch_size {
            sum += *this_inp.add(out_size * i);
        }

        *(out as *mut f32).add(idx) = sum;
    });
}

pub unsafe fn activate_dual(
    _: DeviceHandles,
    batch_size: usize,
    tensor_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    unimplemented!();
}

pub unsafe fn backprop_dual(
    _: DeviceHandles,
    batch_size: usize,
    tensor_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    unimplemented!();
}
