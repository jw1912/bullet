use super::bindings;

pub unsafe fn catch(status: bindings::cudaError_t, name: &str) {
    if status != bindings::cudaError_t::cudaSuccess {
        panic!("{name}: {status:?}");
    }
}

pub fn _device_synchronise() {
    // # Safety
    // This function cannot fail without raising an error
    // that will be caught.
    unsafe {
        catch(bindings::cudaDeviceSynchronize(), "DeviceSynchronize");
    }
}

pub fn _panic_if_device_error(msg: &str) {
    // # Safety
    // This function cannot fail without raising an error
    // that will be caught.
    unsafe {
        catch(bindings::cudaGetLastError(), msg);
    }
}
