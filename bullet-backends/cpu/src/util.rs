/*
The things you have to do for a heterogenous interface...
*/

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};

pub fn device_name() -> String {
    "CPU".to_string()
}

pub fn device_synchronise() {}

pub fn panic_if_device_error(_: &str) {}

pub fn malloc<T>(num: usize) -> *mut T {
    let size = std::mem::size_of::<T>() * num;
    let align = std::mem::align_of::<T>();

    let layout = Layout::from_size_align(size, align).unwrap();

    unsafe {
        let ptr = alloc_zeroed(layout);
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        ptr.cast()
    }
}

/// # Safety
/// Need to make sure not to double free.
pub unsafe fn free(ptr: *mut f32, num: usize) {
    let size = std::mem::size_of::<f32>() * num;
    let align = std::mem::align_of::<f32>();
    let layout = Layout::from_size_align(size, align).unwrap();
    dealloc(ptr.cast(), layout);
}

pub fn calloc<T>(num: usize) -> *mut T {
    malloc(num)
}

pub fn set_zero<T>(ptr: *mut T, num: usize) {
    let byte_ptr = ptr.cast::<u8>();
    let bytes = std::mem::size_of::<T>() * num;
    unsafe {
        for i in 0..bytes {
            *byte_ptr.add(i) = 0;
        }
    }
}

/// # Safety
/// Pointers need to be valid and `amt` need to be valid.
pub unsafe fn copy_to_device<T: Copy>(dest: *mut T, src: *const T, amt: usize) {
    let src_slice = std::slice::from_raw_parts(src, amt);
    let dest_slice = std::slice::from_raw_parts_mut(dest, amt);
    dest_slice.copy_from_slice(src_slice);
}

/// # Safety
/// Pointers need to be valid and `amt` need to be valid.
pub unsafe fn copy_from_device<T: Copy>(dest: *mut T, src: *const T, amt: usize) {
    copy_to_device(dest, src, amt);
}
