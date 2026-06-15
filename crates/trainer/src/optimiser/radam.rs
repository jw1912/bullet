use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
};

use bullet_compiler::tensor::{DType, DValue, IRTrace, TType, TValue};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    runtime::{Device, DeviceProps, Dialect, Dim3, Gpu, Stream},
};

use crate::optimiser::{OptimiserUpdateResult, OptimiserUpdateSync};

use super::{OptimiserState, utils};

#[derive(Clone, Copy, Debug)]
pub struct RAdamParams {
    pub beta1: f32,
    pub beta2: f32,
    pub n_sma_threshold: f32,
    pub decay: f32,
    pub clip: Option<(f32, f32)>,
}

impl Default for RAdamParams {
    fn default() -> Self {
        Self { beta1: 0.9, beta2: 0.999, n_sma_threshold: 5.0, decay: 0.0, clip: None }
    }
}

const OP_CUDA: &str = "\
__device__ __forceinline__ void radamOp(
    const float grad,
    const float rate,
    const int denom,
    float* p,
    float* m,
    float* v
) {
    p[0] *= 1.0F - static_cast<float>(DECAY) * rate;

    m[0] = static_cast<float>(BETA1) * m[0] + (1.0F - static_cast<float>(BETA1)) * grad;
    v[0] = static_cast<float>(BETA2) * v[0] + (1.0F - static_cast<float>(BETA2)) * grad * grad;

    float val = m[0];
    if (denom) val /= sqrt(v[0]) + EPSILON;
    p[0] -= rate * val;

    p[0] = min(max(p[0], static_cast<float>(WMIN)), static_cast<float>(WMAX));
}";

const DECL_CUDA: &str = "
extern \"C\" __global__ void radam(
    const float* adj_ptr,
    const float* rate_ptr,
    const float* step_size_ptr,
    const int* denom_ptr,
    const float* gradients,
    float* network,
    float* momentum,
    float* velocity
)";

const OP_MSL: &str = "\
#include <metal_stdlib>
using namespace metal;

inline void radamOp(
    const float grad,
    const float rate,
    const int denom,
    thread float* p,
    thread float* m,
    thread float* v
) {
    p[0] *= 1.0f - float(DECAY) * rate;

    m[0] = float(BETA1) * m[0] + (1.0f - float(BETA1)) * grad;
    v[0] = float(BETA2) * v[0] + (1.0f - float(BETA2)) * grad * grad;

    float val = m[0];
    if (denom) val /= sqrt(v[0]) + EPSILON;
    p[0] -= rate * val;

    p[0] = min(max(p[0], float(WMIN)), float(WMAX));
}";

const DECL_MSL: &str = "
kernel void radam(
    const device float* adj_ptr [[buffer(0)]],
    const device float* rate_ptr [[buffer(1)]],
    const device float* step_size_ptr [[buffer(2)]],
    const device int* denom_ptr [[buffer(3)]],
    const device float* gradients [[buffer(4)]],
    device float* network [[buffer(5)]],
    device float* momentum [[buffer(6)]],
    device float* velocity [[buffer(7)]],
    uint metal_tid [[thread_position_in_grid]]
)";

impl RAdamParams {
    pub fn build(&self, size: usize, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
        let (min, max) = self.clip.unwrap_or((f32::MIN, f32::MAX));

        let (op_src, decl) = match props.dialect() {
            Dialect::CudaHip => (OP_CUDA, DECL_CUDA),
            Dialect::Msl => (OP_MSL, DECL_MSL),
        };

        let op = op_src
            .replace("DECAY", &format!("{:.E}", self.decay))
            .replace("BETA1", &format!("{:.E}", self.beta1))
            .replace("BETA2", &format!("{:.E}", self.beta2))
            .replace("WMIN", &format!("{min:.E}"))
            .replace("WMAX", &format!("{max:.E}"))
            .replace("EPSILON", "0.00000001F");

        let body = match props.dialect() {
            Dialect::CudaHip => {
                if size.is_multiple_of(4) {
                    format!(
                        "
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid < {})
                {{
                    const float adj = adj_ptr[0];
                    const float rate = rate_ptr[0] * step_size_ptr[0];
                    const int denom = denom_ptr[0];
                    float4 p = ((float4 *)network)[tid];
                    float4 m = ((float4 *)momentum)[tid];
                    float4 v = ((float4 *)velocity)[tid];
                    const float4 g = ((const float4 *)gradients)[tid];

                    radamOp(adj * g.x, rate, denom, &p.x, &m.x, &v.x);
                    radamOp(adj * g.y, rate, denom, &p.y, &m.y, &v.y);
                    radamOp(adj * g.z, rate, denom, &p.z, &m.z, &v.z);
                    radamOp(adj * g.w, rate, denom, &p.w, &m.w, &v.w);

                    ((float4 *)network)[tid] = p;
                    ((float4 *)momentum)[tid] = m;
                    ((float4 *)velocity)[tid] = v;
                }}",
                        size / 4,
                    )
                } else {
                    format!(
                        "
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid < {size})
                {{
                    const float adj = adj_ptr[0];
                    const float rate = rate_ptr[0] * step_size_ptr[0];
                    const int denom = denom_ptr[0];
                    float p = network[tid];
                    float m = momentum[tid];
                    float v = velocity[tid];
                    const float g = gradients[tid];

                    radamOp(adj * g, rate, denom, &p, &m, &v);

                    network[tid] = p;
                    momentum[tid] = m;
                    velocity[tid] = v;
                }}"
                    )
                }
            }
            Dialect::Msl => {
                if size.is_multiple_of(4) {
                    format!(
                        "
                    const uint tid = metal_tid;

                    if (tid < {})
                    {{
                        const float adj = adj_ptr[0];
                        const float rate = rate_ptr[0] * step_size_ptr[0];
                        const int denom = denom_ptr[0];
                        float4 p_vec = ((device float4 *)network)[tid];
                        float4 m_vec = ((device float4 *)momentum)[tid];
                        float4 v_vec = ((device float4 *)velocity)[tid];
                        const float4 g = ((const device float4 *)gradients)[tid];

                        float px = p_vec.x, py = p_vec.y, pz = p_vec.z, pw = p_vec.w;
                        float mx = m_vec.x, my = m_vec.y, mz = m_vec.z, mw = m_vec.w;
                        float vx = v_vec.x, vy = v_vec.y, vz = v_vec.z, vw = v_vec.w;

                        radamOp(adj * g.x, rate, denom, &px, &mx, &vx);
                        radamOp(adj * g.y, rate, denom, &py, &my, &vy);
                        radamOp(adj * g.z, rate, denom, &pz, &mz, &vz);
                        radamOp(adj * g.w, rate, denom, &pw, &mw, &vw);

                        ((device float4 *)network)[tid] = float4(px, py, pz, pw);
                        ((device float4 *)momentum)[tid] = float4(mx, my, mz, mw);
                        ((device float4 *)velocity)[tid] = float4(vx, vy, vz, vw);
                    }}",
                        size / 4,
                    )
                } else {
                    format!(
                        "
                    const uint tid = metal_tid;

                    if (tid < {size})
                    {{
                        const float adj = adj_ptr[0];
                        const float rate = rate_ptr[0] * step_size_ptr[0];
                        const int denom = denom_ptr[0];
                        float p = network[tid];
                        float m = momentum[tid];
                        float v = velocity[tid];
                        const float g = gradients[tid];

                        radamOp(adj * g, rate, denom, &p, &m, &v);

                        network[tid] = p;
                        momentum[tid] = m;
                        velocity[tid] = v;
                    }}"
                    )
                }
            }
        };

        let ty = TType::new(size, DType::F32);
        let sty = TType::new(1, DType::F32);

        let total_threads = if size.is_multiple_of(4) { size / 4 } else { size };
        let src = unsafe {
            KernelSrc::new(
                vec![sty, sty, sty, TType::new(1, DType::I32), ty],
                vec![ty; 3],
                "radam".to_string(),
                format!("{op}{decl}{{{body}}}"),
                vec![(0, true), (1, true), (2, true), (3, true), (4, true), (0, false), (1, false), (2, false)],
                BTreeSet::new(),
                Dim3 { x: total_threads.div_ceil(256) as u32, y: 1, z: 1 },
                256,
                0,
            )
        };

        Ok(src)
    }
}

pub struct RAdam<G: Gpu> {
    momentum: Arc<Buffer<G>>,
    velocity: Arc<Buffer<G>>,
    op: CompiledKernel<G>,
    params: RAdamParams,
    step: usize,
    step_size: Arc<Buffer<G>>,
    denom: Arc<Buffer<G>>,
    cpu_step_size: TValue,
    cpu_denom: TValue,
}

impl<G: Gpu> OptimiserState<G> for RAdam<G> {
    type Params = RAdamParams;

    fn new(device: &Arc<Device<G>>, size: usize, default_params: Self::Params) -> Result<Self, G::Error> {
        let op = default_params.build(size, device.props()).unwrap().compile(device.clone())?;

        Ok(Self {
            momentum: Buffer::from_host(device, &TValue::zeros(DType::F32, size))?,
            velocity: Buffer::from_host(device, &TValue::zeros(DType::F32, size))?,
            op,
            params: default_params,
            step: 0,
            step_size: Buffer::from_host(device, &TValue::zeros(DType::F32, 1))?,
            denom: Buffer::from_host(device, &TValue::zeros(DType::I32, 1))?,
            cpu_step_size: TValue::F32(vec![0.0]),
            cpu_denom: TValue::I32(vec![0]),
        })
    }

    #[allow(unused)]
    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
        assert_eq!(weights.size(), self.momentum.size());
        assert_eq!(weights.size(), self.velocity.size());

        self.step += 1;

        let params = self.params;
        let step = self.step as f32;

        let beta2_t = params.beta2.powf(step);
        let n_sma_max = 2.0 / (1.0 - params.beta2) - 1.0;
        let n_sma = n_sma_max - 2.0 * step * beta2_t / (1.0 - beta2_t);

        let denom = 1.0 - params.beta1.powf(step);
        let step_size = if n_sma > params.n_sma_threshold {
            let p1 = (n_sma - 4.0) / (n_sma_max - 4.0);
            let p2 = (n_sma - 2.0) / n_sma;
            let p3 = n_sma_max / (n_sma_max - 2.0);
            ((1.0 - beta2_t) * p1 * p2 * p3).sqrt() / denom
        } else {
            1.0 / denom
        };

        let denom = i32::from(n_sma > params.n_sma_threshold);

        self.cpu_step_size.write(0, DValue::F32(step_size));
        self.cpu_denom.write(0, DValue::I32(denom));

        let mut sync = OptimiserUpdateSync::default();

        sync.push_copy(self.step_size.copy_from_host_async(stream, &self.cpu_step_size)?);
        sync.push_copy(self.denom.copy_from_host_async(stream, &self.cpu_denom)?);

        sync.push_kernel(self.op.execute(
            stream.clone(),
            vec![gradient_factor, learning_rate, self.step_size.clone(), self.denom.clone(), grads],
            vec![weights, self.momentum.clone(), self.velocity.clone()],
        )?);

        Ok(sync)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        self.step = 0;
        let size = self.momentum.size();
        self.momentum.copy_from_host(&TValue::zeros(DType::F32, size))?;
        self.velocity.copy_from_host(&TValue::zeros(DType::F32, size))?;
        Ok(())
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.momentum)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        utils::write_weights_to_file::<G>(&momentum, &format!("{path}/momentum.bin"))?;
        utils::write_weights_to_file::<G>(&velocity, &format!("{path}/velocity.bin"))?;

        let mut file = File::create(format!("{path}/step.txt")).unwrap();
        for (id, single) in map.iter() {
            writeln!(file, "{id},{}", single.step).unwrap();
        }

        Ok(())
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let paths = [format!("{path}/momentum.bin"), format!("{path}/velocity.bin")];
        let mut momentum = utils::load_weights_from_file(&paths[0]);
        let mut velocity = utils::load_weights_from_file(&paths[1]);

        let file = File::open(format!("{path}/step.txt")).unwrap();
        let mut steps = BufReader::new(file)
            .lines()
            .map(|s| {
                let s = s.unwrap();
                let mut split = s.split(',');
                let id = split.next().unwrap();
                (id.to_string(), split.next().unwrap().parse().unwrap())
            })
            .collect::<Vec<(String, usize)>>();

        momentum.sort_by_key(|(id, _)| id.clone());
        velocity.sort_by_key(|(id, _)| id.clone());
        steps.sort_by_key(|(id, _)| id.clone());

        for (((id1, mom), (id2, vel)), (id3, step)) in momentum.into_iter().zip(velocity).zip(steps) {
            assert_eq!(id1, id2);
            assert_eq!(id1, id3);

            let single = map.get_mut(&id1).unwrap();
            single.momentum.copy_from_host(&TValue::F32(mom))?;
            single.velocity.copy_from_host(&TValue::F32(vel))?;
            single.step = step;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        self.params = params;

        let size = self.momentum.size();
        let device = self.momentum.device();
        self.op = params.build(size, device.props()).unwrap().compile(device)?;
        Ok(())
    }
}
