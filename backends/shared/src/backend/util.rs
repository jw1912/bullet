use super::bindings;

pub unsafe fn catch(status: bindings::cudaError_t, name: &str) {
    if status != bindings::SUCCESS {
        panic!("{name}: {status:?}");
    }
}

pub fn device_synchronise() {
    // # Safety
    // This function cannot fail without raising an error
    // that will be caught.
    unsafe {
        catch(bindings::cudaDeviceSynchronize(), "DeviceSynchronize");
    }
}

pub fn panic_if_device_error(msg: &str) {
    // # Safety
    // This function cannot fail without raising an error
    // that will be caught.
    unsafe {
        catch(bindings::cudaGetLastError(), msg);
    }
}
