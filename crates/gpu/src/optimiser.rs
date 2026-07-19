use bullet_compiler::tensor::{DType, TType};

use crate::{
    kernel::KernelSrc,
    runtime::{Dialect, Dim3},
};

pub fn build_adamw_op(
    size: usize,
    dialect: Dialect,
    decay: f32,
    beta1: f32,
    beta2: f32,
    wmin: f32,
    wmax: f32,
) -> KernelSrc {
    let (op_src, decl) = match dialect {
        Dialect::CudaHip => (CUDA_ADAMW_OP, CUDA_ADAMW_DECL),
        Dialect::Msl => (MSL_ADAMW_OP, MSL_ADAMW_DECL),
    };

    let op = op_src
        .replace("DECAY", &format!("{:.E}", decay))
        .replace("BETA1", &format!("{:.E}", beta1))
        .replace("BETA2", &format!("{:.E}", beta2))
        .replace("WMIN", &format!("{:.E}", wmin))
        .replace("WMAX", &format!("{:.E}", wmax))
        .replace("EPSILON", "0.00000001F");

    let vect = size.is_multiple_of(4);
    let threads = if vect { size / 4 } else { size };

    let body = match dialect {
        Dialect::CudaHip => [CUDA_ADAMW_BODY_SCAL, CUDA_ADAMW_BODY_VECT][usize::from(vect)],
        Dialect::Msl => [MSL_ADAMW_BODY_SCAL, MSL_ADAMW_BODY_VECT][usize::from(vect)],
    };

    let ty = TType::new(size, DType::F32);

    unsafe {
        KernelSrc::new(
            vec![TType::new(1, DType::F32), TType::new(1, DType::F32), ty],
            vec![ty; 3],
            "adamw".to_string(),
            format!("{op}{decl}{{{}}}", body.replace("SIZE", &threads.to_string())),
            vec![(0, true), (1, true), (2, true), (0, false), (1, false), (2, false)],
            Default::default(),
            Dim3 { x: threads.div_ceil(256) as u32, y: 1, z: 1 },
            256,
            0,
        )
    }
}

const CUDA_ADAMW_OP: &str = "\
__device__ __forceinline__ void adamOp(
    const float grad,
    const float rate,
    float* p,
    float* m,
    float* v
) {
    p[0] *= 1.0F - static_cast<float>(DECAY) * rate;

    m[0] = static_cast<float>(BETA1) * m[0] + (1.0F - static_cast<float>(BETA1)) * grad;
    v[0] = static_cast<float>(BETA2) * v[0] + (1.0F - static_cast<float>(BETA2)) * grad * grad;

    float val = m[0] / (sqrtf(v[0]) + static_cast<float>(EPSILON));
    p[0] -= rate * val;

    p[0] = min(max(p[0], static_cast<float>(WMIN)), static_cast<float>(WMAX));
}";

const CUDA_ADAMW_DECL: &str = "
extern \"C\" __global__ void adamw(
    const float* adj_ptr,
    const float* rate_ptr,
    const float* gradients,
    float* network,
    float* momentum,
    float* velocity
)";

const CUDA_ADAMW_BODY_VECT: &str = "
const int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid < SIZE)
{{
    const float adj = adj_ptr[0];
    const float rate = rate_ptr[0];
    float4 p = ((float4 *)network)[tid];
    float4 m = ((float4 *)momentum)[tid];
    float4 v = ((float4 *)velocity)[tid];
    const float4 g = ((const float4 *)gradients)[tid];

    adamOp(adj * g.x, rate, &p.x, &m.x, &v.x);
    adamOp(adj * g.y, rate, &p.y, &m.y, &v.y);
    adamOp(adj * g.z, rate, &p.z, &m.z, &v.z);
    adamOp(adj * g.w, rate, &p.w, &m.w, &v.w);

    ((float4 *)network)[tid] = p;
    ((float4 *)momentum)[tid] = m;
    ((float4 *)velocity)[tid] = v;
}}";

const CUDA_ADAMW_BODY_SCAL: &str = "
const int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid < SIZE)
{{
    const float adj = adj_ptr[0];
    const float rate = rate_ptr[0];
    float p = network[tid];
    float m = momentum[tid];
    float v = velocity[tid];
    const float g = gradients[tid];

    adamOp(adj * g, rate, &p, &m, &v);

    network[tid] = p;
    momentum[tid] = m;
    velocity[tid] = v;
}}";

const MSL_ADAMW_OP: &str = "\
#include <metal_stdlib>
using namespace metal;

inline void adamOp(
    const float grad,
    const float rate,
    thread float* p,
    thread float* m,
    thread float* v
) {
    p[0] *= 1.0f - float(DECAY) * rate;

    m[0] = float(BETA1) * m[0] + (1.0f - float(BETA1)) * grad;
    v[0] = float(BETA2) * v[0] + (1.0f - float(BETA2)) * grad * grad;

    float val = m[0] / (sqrt(v[0]) + float(EPSILON));
    p[0] -= rate * val;

    p[0] = min(max(p[0], float(WMIN)), float(WMAX));
}";

const MSL_ADAMW_DECL: &str = "
kernel void adamw(
    const device float* adj_ptr [[buffer(0)]],
    const device float* rate_ptr [[buffer(1)]],
    const device float* gradients [[buffer(2)]],
    device float* network [[buffer(3)]],
    device float* momentum [[buffer(4)]],
    device float* velocity [[buffer(5)]],
    uint metal_tid [[thread_position_in_grid]]
)";

const MSL_ADAMW_BODY_VECT: &str = "
const uint tid = metal_tid;

if (tid < SIZE)
{{
    const float adj = adj_ptr[0];
    const float rate = rate_ptr[0];
    float4 p_vec = ((device float4 *)network)[tid];
    float4 m_vec = ((device float4 *)momentum)[tid];
    float4 v_vec = ((device float4 *)velocity)[tid];
    const float4 g = ((const device float4 *)gradients)[tid];

    float px = p_vec.x, py = p_vec.y, pz = p_vec.z, pw = p_vec.w;
    float mx = m_vec.x, my = m_vec.y, mz = m_vec.z, mw = m_vec.w;
    float vx = v_vec.x, vy = v_vec.y, vz = v_vec.z, vw = v_vec.w;

    adamOp(adj * g.x, rate, &px, &mx, &vx);
    adamOp(adj * g.y, rate, &py, &my, &vy);
    adamOp(adj * g.z, rate, &pz, &mz, &vz);
    adamOp(adj * g.w, rate, &pw, &mw, &vw);

    ((device float4 *)network)[tid] = float4(px, py, pz, pw);
    ((device float4 *)momentum)[tid] = float4(mx, my, mz, mw);
    ((device float4 *)velocity)[tid] = float4(vx, vy, vz, vw);
}}";

const MSL_ADAMW_BODY_SCAL: &str = "
const uint tid = metal_tid;

if (tid < SIZE)
{{
    const float adj = adj_ptr[0];
    const float rate = rate_ptr[0];
    float p = network[tid];
    float m = momentum[tid];
    float v = velocity[tid];
    const float g = gradients[tid];

    adamOp(adj * g, rate, &p, &m, &v);

    network[tid] = p;
    momentum[tid] = m;
    velocity[tid] = v;
}}";
