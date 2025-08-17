#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif
#include <cstdint>
#include "util.cu"
#include "base/activate.cu"
#include "base/geam.cu"
#include "base/optimiser.cu"
#include "base/pairwise.cu"
#include "base/power_error.cu"
#include "base/scalar.cu"
#include "gather.cu"
#include "select.cu"
#include "softmax.cu"
#include "sparse/fwd.cu"
#include "sparse/bwd.cu"
#include "sparse/to_dense.cu"
