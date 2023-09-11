pub mod bindings;

use std::ffi::{c_float, c_void};

use bindings::{
    cudaDeviceSynchronize,
    cudaError,
    cudaMalloc,
    cudaMemcpy,
    cudaMemcpyKind,
    cudaMemset,
};

#[macro_export]
macro_rules! catch {
    ($func:expr, $caller:expr) => {
        let err = $func;
        if err != cudaError::cudaSuccess {
            panic!("{}: {:?}", $caller, err);
        }
    };
    ($func:expr) => {
        catch!($func, "synchronise")
    }
}

pub fn cuda_malloc<T>(size: usize) -> *mut T {
    let mut grad = std::ptr::null_mut::<T>();

    unsafe {
        let grad_ptr = (&mut grad) as *mut *mut T;
        assert!(!grad_ptr.is_null(), "null pointer");
        catch!(cudaMalloc(grad_ptr.cast(), size), "malloc");
        catch!(cudaDeviceSynchronize());
    }

    grad
}

pub fn cuda_calloc<const SIZE: usize>() -> *mut c_float {
    let mut grad = std::ptr::null_mut::<c_float>();

    unsafe {
        let grad_ptr = (&mut grad) as *mut *mut c_float;
        catch!(cudaMalloc(grad_ptr.cast(), SIZE), "malloc");
        catch!(cudaDeviceSynchronize());
        catch!(cudaMemset(grad as *mut c_void, 0, SIZE), "memset");
        catch!(cudaDeviceSynchronize());
    }

    grad
}

pub fn cuda_copy_to_gpu<T>(dest: *mut T, src: *const T, amt: usize) {
    unsafe {
        catch!(cudaMemcpy(
            dest.cast(),
            src.cast(),
            amt * std::mem::size_of::<T>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice
        ), "memcpy");
        catch!(cudaDeviceSynchronize());
    }
}

#[cfg(test)]
mod test {
    use std::ffi::c_int;
    use super::{*, bindings::*};

    #[test]
    fn test_basic_interop() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [2.0f32, 4.0, 6.0];
        let mut c = [0.0f32; 3];

        let size = std::mem::size_of_val(&a);

        let aptr = cuda_malloc(size);
        let bptr = cuda_malloc(size);
        let cptr = cuda_malloc(size);

        cuda_copy_to_gpu(aptr, a.as_ptr(), 3);
        cuda_copy_to_gpu(bptr, b.as_ptr(), 3);

        unsafe {
            let _ = add(
                aptr,
                bptr,
                cptr,
                size as c_int,
            );

            let _ = cudaMemcpy(
                c.as_mut_ptr().cast(),
                cptr.cast(),
                size,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
        }

        println!("{c:?}");
    }
}