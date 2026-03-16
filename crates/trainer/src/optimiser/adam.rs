use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    sync::Arc,
};

use bullet_compiler::tensor::{DType, IRTrace, TType, TValue};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    runtime::{Dim3, Gpu, Stream},
};

use crate::optimiser::{OptimiserUpdateResult, OptimiserUpdateSync};

use super::{OptimiserState, utils};

#[derive(Clone, Copy, Debug)]
pub struct AdamWParams {
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for AdamWParams {
    fn default() -> Self {
        Self { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 }
    }
}

const OP: &str = "\
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

const DECL: &str = "
extern \"C\" __global__ void adamw(
    const float* adj_ptr,
    const float* rate_ptr,
    const float* gradients,
    float* network,
    float* momentum,
    float* velocity
)";

impl AdamWParams {
    pub fn build(&self, size: usize) -> Result<KernelSrc, IRTrace> {
        let op = OP
            .replace("DECAY", &self.decay.to_string())
            .replace("BETA1", &self.beta1.to_string())
            .replace("BETA2", &self.beta2.to_string())
            .replace("WMIN", &self.min_weight.to_string())
            .replace("WMAX", &self.max_weight.to_string())
            .replace("EPSILON", "0.00000001F");

        let body = if size.is_multiple_of(4) {
            format!(
                "
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid < {})
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
                    const float rate = rate_ptr[0];
                    float p = network[tid];
                    float m = momentum[tid];
                    float v = velocity[tid];
                    const float g = gradients[tid];

                    adamOp(adj * g, rate, &p, &m, &v);

                    network[tid] = p;
                    momentum[tid] = m;
                    velocity[tid] = v;
                }}"
            )
        };

        let ty = TType::new(size, DType::F32);

        let total_threads = if size.is_multiple_of(4) { size / 4 } else { size };
        let src = unsafe {
            KernelSrc::new(
                vec![TType::new(1, DType::F32), TType::new(1, DType::F32), ty],
                vec![ty; 3],
                "adamw".to_string(),
                format!("{op}{DECL}{{{body}}}"),
                false,
                vec![(0, true), (1, true), (2, true), (0, false), (1, false), (2, false)],
                BTreeSet::new(),
                Rc::new(move |_| Dim3 { x: total_threads.div_ceil(256) as u32, y: 1, z: 1 }),
                Rc::new(|_| 256),
                Rc::new(|_| 0),
            )
        };

        Ok(src)
    }
}

pub struct AdamW<G: Gpu> {
    momentum: Arc<Buffer<G>>,
    velocity: Arc<Buffer<G>>,
    op: CompiledKernel<G>,
}

impl<G: Gpu> OptimiserState<G> for AdamW<G> {
    type Params = AdamWParams;

    fn new(stream: &Arc<Stream<G>>, size: usize, default_params: Self::Params) -> Result<Self, G::Error> {
        if default_params.max_weight < default_params.min_weight {
            return Err(
                format!("Invalid clipping: {} >= {}", default_params.min_weight, default_params.max_weight).into()
            );
        }

        let op = default_params.build(size).unwrap().compile(stream.device())?;

        Ok(Self {
            momentum: Buffer::from_host(stream, &TValue::zeros(DType::F32, size))?.value()?.0,
            velocity: Buffer::from_host(stream, &TValue::zeros(DType::F32, size))?.value()?.0,
            op,
        })
    }

    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
        let mut sync = OptimiserUpdateSync::default();

        sync.push_kernel(self.op.execute(
            stream.clone(),
            vec![gradient_factor, learning_rate, grads],
            vec![weights, self.momentum.clone(), self.velocity.clone()],
        )?);

        Ok(sync)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        let stream = self.momentum.creator();
        let size = self.momentum.size();
        self.momentum.copy_from_host(&stream, &TValue::zeros(DType::F32, size))?;
        self.velocity.copy_from_host(&stream, &TValue::zeros(DType::F32, size))?;
        Ok(())
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let stream = map.iter().next().unwrap().1.momentum.creator();

        let momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.momentum)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        utils::write_weights_to_file::<G>(&stream, &momentum, &format!("{path}/momentum.bin"))?;
        utils::write_weights_to_file::<G>(&stream, &velocity, &format!("{path}/velocity.bin"))
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let paths = [format!("{path}/momentum.bin"), format!("{path}/velocity.bin")];
        let mut momentum = utils::load_weights_from_file(&paths[0]);
        let mut velocity = utils::load_weights_from_file(&paths[1]);

        momentum.sort_by_key(|(id, _)| id.clone());
        velocity.sort_by_key(|(id, _)| id.clone());

        for ((id1, mom), (id2, vel)) in momentum.into_iter().zip(velocity) {
            assert_eq!(id1, id2);

            let single = map.get_mut(&id1).unwrap();
            let stream = single.momentum.creator();
            single.momentum.copy_from_host(&stream, &TValue::F32(mom))?;
            single.velocity.copy_from_host(&stream, &TValue::F32(vel))?;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        let size = self.momentum.size();
        let device = self.momentum.creator().device();
        self.op = params.build(size).unwrap().compile(device)?;
        Ok(())
    }
}
