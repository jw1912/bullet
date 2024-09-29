use super::bindings;

pub unsafe fn catch(status: bindings::cudaError_t, name: &str) {
    if status != bindings::cudaError_t::cudaSuccess {
        panic!("{name}: {status:?}");
    }
}

pub unsafe fn device_name() -> String {
    use std::ffi::CStr;
    let mut num = 0;
    catch(bindings::cudaGetDeviceCount(&mut num), "GetDeviceCount");
    assert!(num >= 1);
    let mut props = crate::util::boxed_and_zeroed();
    catch(bindings::cudaGetDeviceProperties_v2(&mut *props, 0), "GetDeviceProperties");

    let mut buf = [0u8; 256];

    for (val, &ch) in buf.iter_mut().zip(props.name.iter()) {
        *val = ch as u8;
    }

    let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
    let my_str = cstr.to_str().unwrap();
    my_str.to_string()
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
