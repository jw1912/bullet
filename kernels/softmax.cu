#include "util.cu"
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif
#include "softmax/masked.cu"
#include "softmax/naive.cu"
