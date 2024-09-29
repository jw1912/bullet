mod bindings;
pub mod ops;
pub mod util;

use bindings::cublasHandle_t;

use crate::tensor::buffer::Buffer;

#[derive(Debug)]
pub struct ExecutionContext {
    handle: cublasHandle_t,
    ones: Buffer<f32>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        let mut handle: cublasHandle_t = std::ptr::null_mut();

        unsafe {
            bindings::cublasCreate_v2((&mut handle) as *mut cublasHandle_t);
        }

        let ones = Buffer::new(1);
        ones.load_from_slice(&[1.0]);

        Self {
            handle,
            ones,
        }
    }
}
