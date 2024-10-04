mod bindings;
mod buffer;
pub mod ops;
pub mod util;

use bindings::cublasHandle_t;
pub use buffer::Buffer;

/// This contains the internal environment for the GPU to use:
/// - BLAS handles
/// - Internal buffers for use in operations without additional allocation overhead
#[derive(Debug)]
pub struct ExecutionContext {
    handle: cublasHandle_t,
    ones: Buffer<f32>,
}

impl Drop for ExecutionContext {
    fn drop(&mut self) {
        unsafe {
            let status = bindings::cublasDestroy_v2(self.handle);
            assert_eq!(status, bindings::CUBLAS_SUCCESS);
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        let mut handle: cublasHandle_t = std::ptr::null_mut();

        unsafe {
            let status = bindings::cublasCreate_v2((&mut handle) as *mut cublasHandle_t);
            assert_eq!(status, bindings::CUBLAS_SUCCESS);
        }

        let ones = Buffer::new(1);
        ones.load_from_slice(&[1.0]);

        Self { handle, ones }
    }
}
