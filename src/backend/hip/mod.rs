#[cfg(feature = "hip")]
mod bindings;
pub mod ops;
pub mod util;

use bindings::hipblasHandle_t;

#[derive(Clone, Copy)]
pub struct DeviceHandles(hipblasHandle_t);

impl std::ops::Deref for DeviceHandles {
    type Target = hipblasHandle_t;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for DeviceHandles {
    fn default() -> Self {
        let mut handle: hipblasHandle_t = std::ptr::null_mut();

        unsafe {
            bindings::hipblasCreate((&mut handle) as *mut hipblasHandle_t);
        }

        Self(handle)
    }
}

impl DeviceHandles {
    pub fn set_threads(&mut self, _: usize) {}
}
