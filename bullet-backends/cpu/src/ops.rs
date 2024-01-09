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

        for i in 0..n {
            *y_ptr.add(i) = 0.0;
        }

        for i in 0..m {
            let x = *x_ptr.add(i);
            let a = a_ptr.add(n * i);
            for j in 0..n {
                *y_ptr.add(j) += *a.add(j) * x;
            }
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

pub unsafe fn reduce_add_mul_vector_vectort(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    y_ptr: *const f32,
    x_ptr: *const f32,
    a_ptr: *mut f32,
    batch_size: usize,
) {
    let a_ptr = a_ptr as usize;
    let x_ptr = x_ptr as usize;
    let y_ptr = y_ptr as usize;

    handle.split_workload(m * n, |_, idx| {
        let j = idx / n;
        let i = idx - j * n;

        let y_ptr = (y_ptr as *const f32).add(i);
        let x_ptr = (x_ptr as *const f32).add(j);

        let mut a = 0.0;

        for k in 0..batch_size {
            a += *y_ptr.add(n * k) * *x_ptr.add(m * k);
        }

        *(a_ptr as *mut f32).add(idx) = a;
    });
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
