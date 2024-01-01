mod bindings;
pub mod ops;
pub mod util;

use bindings::cublasHandle_t;

#[derive(Clone, Copy)]
pub struct CublasHandle(cublasHandle_t);

impl std::ops::Deref for CublasHandle {
    type Target = cublasHandle_t;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for CublasHandle {
    fn default() -> Self {
        let mut handle: cublasHandle_t = std::ptr::null_mut();

        unsafe {
            bindings::cublasCreate_v2((&mut handle) as *mut cublasHandle_t);
        }

        Self(handle)
    }
}
