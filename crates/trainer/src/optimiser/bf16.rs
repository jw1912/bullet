//! Optimiser state (first and second moments) stored as bf16, halving the
//! state's memory traffic in the bandwidth-bound update. Weights, gradients and
//! all arithmetic stay f32: the update reads the moments, casts them to f32,
//! and casts the new values back.
//!
//! The momentum is rounded to nearest. The velocity uses stochastic rounding:
//! with `beta2` close to 1 each step changes it by far less than half a bf16
//! ulp, so round-to-nearest would freeze it; stochastic rounding keeps its
//! expected value exact.

use std::sync::Arc;

use bullet_compiler::tensor::{DType, DValue, TType, TValue, bf16};
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    pointwise::{PointwiseBuf, PointwiseBuilder, PointwiseNode},
    runtime::{Device, Gpu, Stream},
};

/// Per-update random seed in `[0, 1)` for the stochastic rounding, on the device.
pub struct Seed<G: Gpu> {
    buffer: Arc<Buffer<G>>,
    /// The lane offsets `0..4` that [`uniform`] reads in a vectorised kernel,
    /// which has no per-lane index.
    lanes: Option<Arc<Buffer<G>>>,
    cpu: TValue,
    state: u64,
}

impl<G: Gpu> Seed<G> {
    /// For a state of `size` elements (the update is vectorised when `size` is even).
    pub fn new(device: &Arc<Device<G>>, size: usize) -> Result<Self, G::Error> {
        let lanes = vectorised(size).then(|| Buffer::from_host(device, &TValue::F32(vec![0.0, 1.0, 2.0, 3.0])));
        Ok(Self {
            buffer: Buffer::from_host(device, &TValue::zeros(DType::F32, 1))?,
            lanes: lanes.transpose()?,
            cpu: TValue::F32(vec![0.0]),
            state: 0x9e37_79b9_7f4a_7c15,
        })
    }

    /// The kernel inputs, in the order of [`seed_buffers`].
    pub fn inputs(&self) -> Vec<Arc<Buffer<G>>> {
        std::iter::once(self.buffer.clone()).chain(self.lanes.clone()).collect()
    }

    /// Draws the next seed and queues its copy to the device.
    pub fn advance<'a>(&'a mut self, stream: &Arc<Stream<G>>) -> Result<SyncOnValue<G, &'a TValue>, G::Error> {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.cpu.write(0, DValue::F32((self.state >> 40) as f32 / (1u64 << 24) as f32));
        self.buffer.copy_from_host_async(stream, &self.cpu)
    }
}

/// Whether the update of `size` elements is vectorised (`PointwiseBuilder::vectorised`).
fn vectorised(size: usize) -> bool {
    size.trailing_zeros() > 0
}

/// The kernel arguments [`uniform`] reads: the seed and, when vectorised,
/// the lane offsets, matching [`Seed::inputs`].
pub fn seed_buffers(builder: &PointwiseBuilder, p2size: u8) -> (PointwiseBuf<'_>, Option<PointwiseBuf<'_>>) {
    let seed = builder.new_buffer(TType::new(1, DType::F32));
    (seed, (p2size > 0).then(|| builder.new_buffer(TType::new(4, DType::F32))))
}

/// A state value read from its buffer, as f32.
pub fn to_f32(x: PointwiseNode<'_>) -> PointwiseNode<'_> {
    if x.dtype() == DType::F32 { x } else { x.cast(DType::F32) }
}

/// Per-element uniform numbers in `[0, 1)`: the per-update seed plus a fixed
/// offset per element (multiples of the golden ratio, by thread index and
/// lane), modulo 1. A uniform seed shifted by any fixed offset is still
/// uniform modulo 1, so every element's number is uniform and fresh each
/// update, and the offsets decorrelate neighbouring elements. The thread
/// index is reduced modulo 2^16 first so the float product stays exact
/// enough to spread the offsets.
pub fn uniform<'a>(
    builder: &'a PointwiseBuilder,
    seed_buf: PointwiseBuf<'a>,
    lanes_buf: Option<PointwiseBuf<'a>>,
    p2size: u8,
) -> PointwiseNode<'a> {
    let offset = builder.tid().rem(1 << 16).cast(DType::F32) * 0.618_034 + seed_buf.read(0, 0);
    let u = match lanes_buf {
        Some(lanes) => offset.splat(p2size) + lanes.read(0, p2size) * 0.381_966,
        None => offset,
    };
    u - u.truncate()
}

/// Round `x` (non-negative f32) to bf16 stochastically with uniform `u`: up to
/// the next bf16 above with probability equal to the distance from the one
/// below, as a fraction of their gap. The neighbours come from round-to-nearest
/// casts of slightly scaled values, which land on the adjacent bf16 exactly
/// (the scale moves the value between 0.6 and 1.25 of a gap, away from ties).
pub fn round_stochastic<'a>(x: PointwiseNode<'a>, u: PointwiseNode<'a>) -> PointwiseNode<'a> {
    let rn = |v: PointwiseNode<'a>| v.cast(DType::Bf16).cast(DType::F32);
    let a = rn(x);
    let below = rn(a * (1.0 - 1.25 / 256.0));
    let lo = a + (a - x).is_positive() * (below - a);
    let hi = rn(lo * (1.0 + 0.75 / 128.0));
    let gap = hi - lo;
    let p = (x - lo) * gap.max(f32::MIN_POSITIVE).recip();
    (lo + (p - u).is_positive() * gap).cast(DType::Bf16)
}

/// The type of a state buffer.
pub fn state_type(size: usize, bf16: bool) -> TType {
    TType::new(size, if bf16 { DType::Bf16 } else { DType::F32 })
}

/// Zeroed state.
pub fn zeros(size: usize, bf16: bool) -> TValue {
    TValue::zeros(if bf16 { DType::Bf16 } else { DType::F32 }, size)
}

/// The state as an f32 buffer (a converted copy when bf16), for checkpoints.
pub fn as_f32<G: Gpu>(buf: &Arc<Buffer<G>>) -> Result<Arc<Buffer<G>>, G::Error> {
    match buf.to_host()? {
        TValue::Bf16(v) => Buffer::from_host(&buf.device(), &TValue::F32(v.iter().map(|x| x.to_f32()).collect())),
        _ => Ok(buf.clone()),
    }
}

/// f32 values (from a checkpoint) in the state's storage.
pub fn from_f32(values: Vec<f32>, bf16: bool) -> TValue {
    if bf16 {
        TValue::Bf16(values.into_iter().map(bf16::from_f32).collect())
    } else {
        TValue::F32(values)
    }
}

/// The state in the other storage (`set_params` switching `bf16_state` on an
/// existing optimiser), keeping its values.
pub fn convert<G: Gpu>(buf: &Arc<Buffer<G>>, to_bf16: bool) -> Result<Arc<Buffer<G>>, G::Error> {
    let TValue::F32(values) = as_f32(buf)?.to_host()? else { unreachable!("f32 copy") };
    Buffer::from_host(&buf.device(), &from_f32(values, to_bf16))
}

#[cfg(test)]
mod tests {
    #![cfg_attr(not(any(feature = "cuda", feature = "rocm", feature = "metal")), allow(unused_imports))]
    use super::*;
    use crate::optimiser::{
        OptimiserState,
        adam::{AdamW, AdamWParams},
        radam::{RAdam, RAdamParams},
    };

    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    /// Runs the same random gradient stream through an f32-state and a bf16-state
    /// optimiser for `steps` updates and returns (||w_bf16 - w_f32|| / ||w_f32 - w_0||,
    /// sum |v_bf16 - v_f32| / sum v_f32).
    fn divergence<G: Gpu, S: OptimiserState<G>>(
        params: impl Fn(bool) -> S::Params,
        velocity: impl Fn(&S) -> Result<Vec<f32>, G::Error>,
        steps: usize,
        n: usize,
    ) -> Result<(f32, f32), G::Error> {
        let device = Device::<G>::new(0)?;
        let stream = device.new_stream()?;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let w0: Vec<f32> = (0..n).map(|_| next()).collect();
        // A per-weight gradient bias, so the velocity settles to a steady level.
        let bias: Vec<f32> = (0..n).map(|_| 0.01 * next()).collect();

        let mut f32_state = S::new(&device, n, params(false))?;
        // Created with f32 state and switched by set_params, as the trainer API does it.
        let mut bf16_state = S::new(&device, n, params(false))?;
        bf16_state.set_params(params(true))?;
        let w_f32 = Buffer::from_host(&device, &TValue::F32(w0.clone()))?;
        let w_bf16 = Buffer::from_host(&device, &TValue::F32(w0.clone()))?;
        let adj = Buffer::from_host(&device, &TValue::F32(vec![1.0]))?;
        let lr = Buffer::from_host(&device, &TValue::F32(vec![1e-3]))?;

        for _ in 0..steps {
            let g: Vec<f32> = bias.iter().map(|b| b + 0.005 * next()).collect();
            let grads = Buffer::from_host(&device, &TValue::F32(g))?;
            f32_state.update(&stream, w_f32.clone(), grads.clone(), adj.clone(), lr.clone())?.sync()?;
            bf16_state.update(&stream, w_bf16.clone(), grads, adj.clone(), lr.clone())?.sync()?;
        }

        let (TValue::F32(a), TValue::F32(b)) = (w_f32.to_host()?, w_bf16.to_host()?) else { unreachable!() };
        let moved: f32 = a.iter().zip(&w0).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        let (vf, vb) = (velocity(&f32_state)?, velocity(&bf16_state)?);
        let verr = vf.iter().zip(&vb).map(|(x, y)| (x - y).abs()).sum::<f32>() / vf.iter().sum::<f32>();
        Ok((diff / moved, verr))
    }

    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    fn matches_f32<G: Gpu>() -> Result<(), G::Error> {
        // Long enough for the velocity to settle, where round-to-nearest would
        // stop it moving (each step changes it by ~0.1%, under half a bf16 ulp).
        // 4096 is vectorised four wide, 4099 not at all.
        let steps = 3000;
        for n in [4096, 4099] {
            let (wa, va) = divergence::<G, AdamW<G>>(
                |bf16_state| AdamWParams { bf16_state, ..Default::default() },
                AdamW::velocity_f32,
                steps,
                n,
            )?;
            let (wr, vr) = divergence::<G, RAdam<G>>(
                |bf16_state| RAdamParams { bf16_state, ..Default::default() },
                RAdam::velocity_f32,
                steps,
                n,
            )?;
            println!("n {n}: AdamW weights {wa} velocity {va}; RAdam weights {wr} velocity {vr}");
            assert!(
                wa < WEIGHT_TOLERANCE && va < VELOCITY_TOLERANCE,
                "AdamW bf16 state drifted from f32 (n {n}): weights {wa}, velocity {va}"
            );
            assert!(
                wr < WEIGHT_TOLERANCE && vr < VELOCITY_TOLERANCE,
                "RAdam bf16 state drifted from f32 (n {n}): weights {wr}, velocity {vr}"
            );
        }
        Ok(())
    }

    // Measured on CUDA with the earlier hand-written kernel: stochastic rounding gives
    // ~3% elementwise velocity error and weight divergence up to ~0.011 (RAdam);
    // round-to-nearest gives ~28% and ~0.12.
    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    const VELOCITY_TOLERANCE: f32 = 0.06;
    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    const WEIGHT_TOLERANCE: f32 = 0.03;

    #[cfg(feature = "cuda")]
    mod cuda {
        use bullet_gpu::runtime::cuda::{Cuda, CudaError};

        #[test]
        fn bf16_state_matches_f32() -> Result<(), CudaError> {
            super::matches_f32::<Cuda>()
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use bullet_gpu::runtime::rocm::{ROCm, ROCmError};

        #[test]
        fn bf16_state_matches_f32() -> Result<(), ROCmError> {
            super::matches_f32::<ROCm>()
        }
    }

    #[cfg(feature = "metal")]
    mod metal {
        use bullet_gpu::runtime::metal::{Metal, MetalError};

        #[test]
        fn bf16_state_matches_f32() -> Result<(), MetalError> {
            super::matches_f32::<Metal>()
        }
    }
}
