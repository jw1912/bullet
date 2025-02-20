use crate::DeviceError;

use super::bindings;

pub unsafe fn catch(status: bindings::cudaError_t) -> Result<(), DeviceError> {
    Result::from(status)
}

pub unsafe fn catch_cublas(status: bindings::cublasStatus_t) -> Result<(), DeviceError> {
    Result::from(status)
}

pub fn device_synchronise() -> Result<(), DeviceError> {
    // # Safety
    // This function cannot fail without raising an error
    // that will be caught.
    unsafe { catch(bindings::cudaDeviceSynchronize()) }
}

pub fn get_last_error() -> Result<(), DeviceError> {
    // # Safety
    // This function cannot fail without raising an error
    // that will be caught.
    unsafe { catch(bindings::cudaGetLastError()) }
}
