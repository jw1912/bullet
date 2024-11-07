#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif
#include "sparse/bwd.cu"
#include "sparse/dual_fwd.cu"
#include "sparse/dual_bwd.cu"
#include "sparse/fwd.cu"
