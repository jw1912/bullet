/*
This file exists to generate HIP bindings for Rust, via bindgen.
*/

#ifndef WRAPPER
#define WRAPPER

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#endif