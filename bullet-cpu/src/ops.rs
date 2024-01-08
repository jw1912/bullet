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
    unimplemented!();
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
    unimplemented!();
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
    unimplemented!();
}

pub unsafe fn reduce_add(
    handle: DeviceHandles,
    ones: *const f32,
    batch_size: usize,
    out_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    unimplemented!();
}
