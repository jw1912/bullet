use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use bullet_compiler::{
    ir::IRError,
    tensor::{DType, TType, TValue},
};
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseBuilder,
    runtime::{Device, DeviceProps, Event, Gpu, Stream},
};

use crate::run::logger;

use super::{
    drain,
    plan::{Op, Plan, Slot, shard_ranges},
    release_all,
};

const PROBE_SIZES: [usize; 3] = [1 << 14, 1 << 18, 1 << 21];
const PROBE_MAX: usize = PROBE_SIZES[PROBE_SIZES.len() - 1];
const PROBE_REPEATS: usize = 3;

static PEER_FAILED: AtomicBool = AtomicBool::new(false);

pub struct Comm<G: Gpu> {
    plan: Plan,
    devices: Vec<Arc<Device<G>>>,
    sizes: Vec<(DType, usize)>,
    shards: Vec<Vec<Option<Shard<G>>>>,
    events: Vec<Event<G>>,
}

struct Shard<G: Gpu> {
    stage: Vec<Option<Arc<Buffer<G>>>>,
    reduced: Arc<Buffer<G>>,
    sum: CompiledKernel<G>,
}

impl<G: Gpu> Comm<G> {
    pub fn new(devices: &[Arc<Device<G>>], sizes: &[(DType, usize)]) -> Result<Self, G::Error> {
        let ranks = devices.len();

        for src in devices {
            for dst in devices.iter().filter(|dst| dst.ordinal() != src.ordinal()) {
                verify_peer_access(src, dst)?;
            }
        }

        if PEER_FAILED.load(Ordering::Relaxed) {
            for src in devices {
                for dst in devices {
                    src.disable_peer_access(dst)?;
                }
            }
        }

        let mut shards = Vec::new();

        for (rank, device) in devices.iter().enumerate() {
            let mut tensors = Vec::new();

            for &(dtype, size) in sizes {
                let range = shard_ranges(size, ranks)[rank].clone();
                let active = ranks > 1 && !range.is_empty();
                tensors.push(if active { Some(Shard::new(device, dtype, size, range, ranks, rank)?) } else { None });
            }

            shards.push(tensors);
        }

        let tensor_sizes = sizes.iter().map(|&(_, size)| size).collect::<Vec<_>>();
        let plan = Plan::all_reduce(ranks, &tensor_sizes);

        let events = devices.iter().map(|device| Event::new(device.clone())).collect::<Result<_, _>>()?;

        Ok(Self { plan, devices: devices.to_vec(), sizes: sizes.to_vec(), shards, events })
    }

    pub fn all_reduce(&self, streams: &[Arc<Stream<G>>], bufs: &[Vec<Arc<Buffer<G>>>]) -> Result<(), G::Error> {
        self.validate(streams, bufs)?;
        self.run(&self.plan, streams, bufs)
    }

    pub fn broadcast(
        &self,
        root: usize,
        streams: &[Arc<Stream<G>>],
        bufs: &[Vec<Arc<Buffer<G>>>],
    ) -> Result<(), G::Error> {
        self.validate(streams, bufs)?;
        let sizes = self.sizes.iter().map(|&(_, size)| size).collect::<Vec<_>>();
        self.run(&Plan::broadcast(self.devices.len(), root, &sizes), streams, bufs)
    }

    fn buffer<'a>(
        &'a self,
        rank: usize,
        tensor: usize,
        slot: Slot,
        bufs: &'a [Vec<Arc<Buffer<G>>>],
    ) -> &'a Arc<Buffer<G>> {
        let shard = || self.shards[rank][tensor].as_ref().unwrap();

        match slot {
            Slot::Data => &bufs[rank][tensor],
            Slot::Stage(peer) => shard().stage[peer].as_ref().unwrap(),
            Slot::Reduced => &shard().reduced,
        }
    }

    fn reduce(
        &self,
        rank: usize,
        tensor: usize,
        streams: &[Arc<Stream<G>>],
        bufs: &[Vec<Arc<Buffer<G>>>],
    ) -> Result<SyncOnValue<G, &CompiledKernel<G>>, G::Error> {
        let shard = self.shards[rank][tensor].as_ref().unwrap();
        let inputs = shard.stage.iter().flatten().cloned().collect();
        shard.sum.execute(streams[rank].clone(), inputs, vec![bufs[rank][tensor].clone(), shard.reduced.clone()])
    }

    fn barrier(&self, streams: &[Arc<Stream<G>>]) -> Result<(), G::Error> {
        for (stream, event) in streams.iter().zip(&self.events) {
            stream.record(event)?;
        }

        for (rank, stream) in streams.iter().enumerate() {
            for (_, event) in self.events.iter().enumerate().filter(|&(peer, _)| peer != rank) {
                stream.wait(event)?;
            }
        }

        Ok(())
    }

    fn run(&self, plan: &Plan, streams: &[Arc<Stream<G>>], bufs: &[Vec<Arc<Buffer<G>>>]) -> Result<(), G::Error> {
        if plan.rounds.is_empty() {
            return Ok(());
        }

        drain(self.enqueue(plan, streams, bufs), streams)
    }

    fn enqueue(&self, plan: &Plan, streams: &[Arc<Stream<G>>], bufs: &[Vec<Arc<Buffer<G>>>]) -> Result<(), G::Error> {
        self.barrier(streams)?;

        for round in &plan.rounds {
            let (mut copies, mut sums) = (Vec::new(), Vec::new());

            for op in round {
                match *op {
                    Op::Copy { tensor, src, src_start, dst, dst_start, len } => {
                        let (from, to) =
                            (self.buffer(src.0, tensor, src.1, bufs), self.buffer(dst.0, tensor, dst.1, bufs));
                        let copy = to.copy_range_from_async(&streams[src.0], dst_start, from, src_start, len)?;
                        copies.push(SyncOnValue::new(copy, ()));
                    }
                    Op::Reduce { rank, tensor } => sums.push(self.reduce(rank, tensor, streams, bufs)?),
                }
            }

            unsafe {
                release_all(copies);
                release_all(sums);
            }

            self.barrier(streams)?;
        }

        Ok(())
    }

    fn validate(&self, streams: &[Arc<Stream<G>>], bufs: &[Vec<Arc<Buffer<G>>>]) -> Result<(), G::Error> {
        let (ranks, tensors) = (self.devices.len(), self.sizes.len());

        if streams.len() != ranks || bufs.len() != ranks || bufs.iter().any(|bufs| bufs.len() != tensors) {
            return Err(format!("Comm expects {ranks} ranks of {tensors} tensors!").into());
        }

        for (rank, (device, stream)) in self.devices.iter().zip(streams).enumerate() {
            if stream.device().ordinal() != device.ordinal() {
                return Err(format!("Comm stream mismatch at rank {rank}!").into());
            }

            for (tensor, (buf, &(dtype, size))) in bufs[rank].iter().zip(&self.sizes).enumerate() {
                if buf.device().ordinal() != device.ordinal() || buf.dtype() != dtype || buf.size() != size {
                    return Err(format!("Comm buffer mismatch at rank {rank}, tensor {tensor}!").into());
                }
            }
        }

        Ok(())
    }
}

impl<G: Gpu> Shard<G> {
    fn new(
        device: &Arc<Device<G>>,
        dtype: DType,
        total: usize,
        range: Range<usize>,
        ranks: usize,
        rank: usize,
    ) -> Result<Self, G::Error> {
        let len = range.len();
        let stage = (0..ranks).map(|peer| (peer != rank).then(|| Buffer::zeroed(device, dtype, len)).transpose());

        Ok(Self {
            stage: stage.collect::<Result<_, _>>()?,
            reduced: Buffer::zeroed(device, dtype, len)?,
            sum: build_sum(dtype, total, range, ranks, rank, device.props()).unwrap().compile(device.clone())?,
        })
    }
}

fn verify_peer_access<G: Gpu>(src: &Arc<Device<G>>, dst: &Arc<Device<G>>) -> Result<(), G::Error> {
    if PEER_FAILED.load(Ordering::Relaxed) || !src.enable_peer_access(dst)? {
        return src.disable_peer_access(dst);
    }

    if probe_copies(src, dst)? {
        return Ok(());
    }

    src.disable_peer_access(dst)?;

    let (a, b) = (src.ordinal(), dst.ordinal());
    if !PEER_FAILED.swap(true, Ordering::Relaxed) {
        let warning = format!("Warning: peer access GPU{a} -> GPU{b} returned wrong data, using staged copies instead");
        println!("{}", logger::ansi(warning, 31));
    }

    if !probe_copies(src, dst)? {
        return Err(format!("Copies from GPU{a} to GPU{b} return wrong data!").into());
    }

    Ok(())
}

fn probe_copies<G: Gpu>(src: &Arc<Device<G>>, dst: &Arc<Device<G>>) -> Result<bool, G::Error> {
    let seed = (src.ordinal() << 8) | dst.ordinal();
    let pattern = (0..(PROBE_MAX + PROBE_REPEATS) as i32).map(|i| i ^ seed).collect::<Vec<_>>();

    let stream = src.new_stream()?;
    let from = Buffer::from_host(src, &TValue::I32(pattern.clone()))?;

    for size in PROBE_SIZES {
        let to = Buffer::zeroed(dst, DType::I32, size)?;

        for repeat in 0..PROBE_REPEATS {
            to.copy_range_from_async(&stream, 0, &from, repeat, size)?.sync()?;

            if !matches!(to.to_host()?, TValue::I32(values) if values == pattern[repeat..repeat + size]) {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

fn build_sum(
    dtype: DType,
    total: usize,
    range: Range<usize>,
    ranks: usize,
    rank: usize,
    props: &DeviceProps,
) -> Result<KernelSrc, IRError> {
    let len = range.len();
    let (builder, p2size) = PointwiseBuilder::vectorised(len);

    let data = builder.new_buffer(TType::new(total, dtype));
    let mut stages = (1..ranks).map(|_| builder.new_buffer(TType::new(len, dtype))).collect::<Vec<_>>().into_iter();
    let reduced = builder.new_buffer(TType::new(len, dtype));

    let tid = builder.tid();
    let idx = tid + (range.start >> p2size) as i32;

    let values = (0..ranks)
        .map(|peer| if peer == rank { data.read(idx, p2size) } else { stages.next().unwrap().read(tid, p2size) });
    let sum = values.reduce(|a, b| a + b).unwrap();

    data.write(idx, sum);
    reduced.write(tid, sum);

    unsafe { builder.inner().lower("comm_sum".to_string(), props) }
}
