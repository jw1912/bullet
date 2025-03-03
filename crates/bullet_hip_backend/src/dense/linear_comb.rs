use bullet_core::backend::device::DeviceBuffer;

use crate::{
    backend::{blas, ops, util::catch_cublas},
    Buffer, DeviceError,
};

pub fn linear_comb_single(
    size: usize,
    alpha: f32,
    input_a: Option<&Buffer<f32>>,
    beta: f32,
    input_b: Option<&Buffer<f32>>,
    output: &mut Buffer<f32>,
) -> Result<(), DeviceError> {
    // cublas scale is super slow for some reason
    if let (None, None) = (input_a, input_b) {
        return scale(size, output, alpha);
    }

    let aptr = input_a.map(|a| {
        assert!(size <= a.size());
        a.ptr()
    });

    let bptr = if let Some(b) = input_b {
        assert!(size <= b.size());
        b.ptr()
    } else {
        std::ptr::null()
    };

    unsafe {
        let err =
            blas::linear_comb_matrices(output.device().as_ref(), size, 1, alpha, aptr, beta, bptr, output.mut_ptr());
        catch_cublas(err)
    }
}

fn scale(size: usize, params: &mut Buffer<f32>, alpha: f32) -> Result<(), DeviceError> {
    if size > params.size() {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    unsafe {
        ops::scale(size, params.mut_ptr(), alpha);
    }

    Ok(())
}
