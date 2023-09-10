#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]
#![allow(missing_debug_implementations)]
#![allow(improper_ctypes)]

use std::ffi::{c_void, c_float, c_int};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub const CU_LAUNCH_PARAM_END: *mut c_void = 0 as *mut c_void;
pub const CU_LAUNCH_PARAM_BUFFER_POINTER: *mut c_void = 1 as *mut c_void;
pub const CU_LAUNCH_PARAM_BUFFER_SIZE: *mut c_void = 2 as *mut c_void;

#[link(name = "kernels", kind = "static")]
extern "C" {
    pub fn add(
        a: *const c_float,
        b: *const c_float,
        c: *mut c_float,
        size: c_int,
    ) -> cudaError_t;
}
