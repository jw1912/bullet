#![allow(unused_variables, clippy::missing_safety_doc, clippy::too_many_arguments)]
mod backprops;
mod bufops;
mod mpe;
mod sparse_affine;
mod splat_add;
mod update;

use super::{util, DeviceHandles};

pub use backprops::*;
pub use bufops::*;
pub use mpe::*;
pub use sparse_affine::*;
pub use splat_add::*;
pub use update::*;

pub unsafe fn splat_mul_matrix_vector(
    handle: &DeviceHandles,
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
    handle: &DeviceHandles,
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
    handle: &DeviceHandles,
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
    handle: &DeviceHandles,
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

pub unsafe fn select(
    _: &DeviceHandles,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    buckets: *const u8,
    inp: *const f32,
    out: *mut f32,
) {
    unimplemented!();
}

pub unsafe fn select_backprop(
    _: &DeviceHandles,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    buckets: *const u8,
    inp: *const f32,
    out: *mut f32,
) {
    unimplemented!();
}

pub unsafe fn pairwise_mul(
    handle: &DeviceHandles,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    let inp_addr = inp as usize;
    let out_addr = out as usize;
    handle.split_workload(output_size, |_, idx| {
        let a = *(inp_addr as *const f32).add(idx);
        let b = *(inp_addr as *const f32).add(idx).add(output_size);

        *(out_addr as *mut f32).add(idx) = a * b;
    });
}

pub unsafe fn backprop_pairwise_mul(
    handle: &DeviceHandles,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    let inp_addr = inp as usize;
    let out_addr = out as usize;
    handle.split_workload(input_size, |_, idx| {
        let val_left = *(out_addr as *const f32).add(idx);
        let val_right = *(out_addr as *const f32).add(idx).add(input_size);

        // get the value of the incoming gradient on the output neuron
        let grad_in = *(inp_addr as *const f32).add(idx);

        let grad_left = grad_in * val_right;
        let grad_right = grad_in * val_left;

        *(out_addr as *mut f32).add(idx) = grad_left;
        *(out_addr as *mut f32).add(idx).add(input_size) = grad_right;
    });
}
