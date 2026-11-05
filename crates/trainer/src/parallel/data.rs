use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
    ops::Range,
    sync::Arc,
};

use bullet_compiler::{
    ir::{IRError, NodeId},
    model::{
        Layout,
        operations::{Dim, Reduce},
    },
    tensor::{DType, TType, TValue, TensorIR, operation::Reduction},
};
use bullet_gpu::{
    buffer::{Buffer, PinnedBuffer, SyncOnValue},
    function::Function,
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseBuilder,
    runtime::{Device, DeviceProps, Gpu, Stream},
};

use crate::{
    model::{ModelDefinition, ModelFunctionDefinition, ModelWeights, TensorMap},
    optimiser::{AdditionalUpdate, OptimiserState, OptimiserUpdateResult, OptimiserUpdateSync},
    run::{PreparedBatchHost, TrainingError},
};

use super::{comm::Comm, drain, load_state, plan::split_batch, release_all, sync_all, write_state};

const NOT_PREPARED: &str = "Data-parallel training not prepared!";

pub struct StepOutput {
    pub loss: f32,
    pub grad_sqnorms: Option<Vec<(String, f32)>>,
    pub next_queued: bool,
}

pub struct DataParallel<G: Gpu, S: OptimiserState<G>> {
    replicas: Vec<Replica<G, S>>,
    comm: Comm<G>,
    run: Option<RunState<G>>,
}

struct Replica<G: Gpu, S> {
    device: Arc<Device<G>>,
    compute: Arc<Stream<G>>,
    copy: Arc<Stream<G>>,
    weights: TensorMap<G>,
    state: BTreeMap<String, S>,
}

struct Bound<G: Gpu> {
    tensors: BTreeMap<NodeId, Arc<Buffer<G>>>,
    grads: TensorMap<G>,
    batch: TensorMap<G>,
    next: TensorMap<G>,
    loss: Arc<Buffer<G>>,
    inputs: Vec<RankInput>,
    csr: BTreeMap<String, Csr<G>>,
    host_batch: usize,
}

struct Csr<G: Gpu> {
    offsets: Arc<Buffer<G>>,
    indices: Arc<Buffer<G>>,
    expand: CompiledKernel<G>,
}

struct Upload<'a, G: Gpu> {
    copies: Vec<SyncOnValue<G, &'a PinnedBuffer<G>>>,
    expands: Vec<SyncOnValue<G, &'a CompiledKernel<G>>>,
}

struct RankRun<G: Gpu> {
    func: Function<G>,
    bound: Bound<G>,
    rows: Range<usize>,
    tgf: Arc<Buffer<G>>,
    tlr: Arc<Buffer<G>>,
}

struct RankInput {
    node: NodeId,
    name: String,
    stride: usize,
    host_size: usize,
}

struct NormRun<G: Gpu> {
    names: Vec<String>,
    sqnorms: Arc<Buffer<G>>,
    func: Option<Function<G>>,
    tensors: BTreeMap<NodeId, Arc<Buffer<G>>>,
}

struct RunState<G: Gpu> {
    ranks: Vec<RankRun<G>>,
    norms: Option<NormRun<G>>,
}

impl<G: Gpu, S: OptimiserState<G>> DataParallel<G, S> {
    pub fn new(
        definition: &ModelDefinition,
        weights: &ModelWeights,
        devices: &[Arc<Device<G>>],
        params: S::Params,
    ) -> Result<Self, G::Error> {
        check_devices(devices)?;

        if devices.len() > 1 && !is_summed_over_batch(definition) {
            return Err("Data-parallel training requires a loss summed over the batch!".to_string().into());
        }

        let sizes = weights.iter().map(|(_, value)| (value.values.dtype(), value.values.size())).collect::<Vec<_>>();
        let comm = Comm::new(devices, &sizes)?;

        let replicas = devices.iter().map(|device| Replica::new(device, weights.to_device(device)?, params.clone()));
        let replicas = replicas.collect::<Result<_, _>>()?;

        Ok(Self { replicas, comm, run: None })
    }

    pub fn device(&self) -> Arc<Device<G>> {
        self.replicas[0].device.clone()
    }

    pub fn ranks(&self) -> usize {
        self.replicas.len()
    }

    pub fn weights(&self) -> &TensorMap<G> {
        &self.replicas[0].weights
    }

    pub fn state(&self) -> &BTreeMap<String, S> {
        &self.replicas[0].state
    }

    pub fn states_mut(&mut self) -> impl Iterator<Item = &mut BTreeMap<String, S>> {
        self.replicas.iter_mut().map(|replica| &mut replica.state)
    }

    pub fn read_weights(&self, host: &mut ModelWeights) -> Result<(), G::Error> {
        host.load_from_device(self.weights())
    }

    pub fn write_state(&self, path: &str) -> Result<(), G::Error> {
        write_state(self.state(), path)
    }

    pub fn load_state(&mut self, path: &str) -> Result<(), G::Error> {
        self.states_mut().try_for_each(|state| load_state(state, path))
    }

    pub fn write_weights(&self, host: &ModelWeights) -> Result<(), G::Error> {
        host.write_to_device(self.weights())?;

        let streams = self.replicas.iter().map(|replica| replica.compute.clone()).collect::<Vec<_>>();
        let bufs = self.replicas.iter().map(|replica| replica.weights.values().cloned().collect()).collect::<Vec<_>>();
        self.comm.broadcast(0, &streams, &bufs)
    }

    pub fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
        gradients: &TensorMap<G>,
        pre_update: &'a mut [Box<dyn AdditionalUpdate<G>>],
        post_update: &'a mut [Box<dyn AdditionalUpdate<G>>],
    ) -> OptimiserUpdateResult<'a, G> {
        let replica = &mut self.replicas[0];
        update_replica(replica, pre_update, post_update, stream, gradient_factor, learning_rate, gradients)
    }

    pub fn prepare(
        &mut self,
        defn: &ModelDefinition,
        batch_size: usize,
        grad_norms: bool,
    ) -> Result<(), TrainingError<G>> {
        self.run = None;

        let ranks = self.ranks();
        if ranks > batch_size {
            return Err(TrainingError::MoreDevicesThanBatchSize);
        }

        let rows = split_batch(batch_size, ranks);

        let lens = rows.iter().map(Range::len).collect::<BTreeSet<_>>();
        let mut lowered = BTreeMap::new();
        for len in lens {
            let lowering = defn.lower_backward(&Default::default(), len, grad_norms && ranks == 1, false);
            lowered.insert(len, lowering.map_err(TrainingError::CompilingBackwards)?);
        }

        let mut run = RunState { ranks: Vec::new(), norms: None };

        for (replica, rows) in self.replicas.iter().zip(rows) {
            let (func, gmap, norms) = &lowered[&rows.len()];
            let mut rank = RankRun::new(defn, replica, func, gmap, rows, batch_size)?;

            // squared gradient norms
            if let Some(norms) = norms {
                let sqnorms = Buffer::zeroed(&replica.device, DType::F32, norms.names.len())
                    .map_err(TrainingError::Unexpected)?;
                rank.bound.tensors.insert(norms.acc, sqnorms.clone());
                run.norms = Some(NormRun { names: norms.names.clone(), sqnorms, func: None, tensors: BTreeMap::new() });
            }

            run.ranks.push(rank);
        }

        if grad_norms && ranks > 1 {
            let device = &self.device();
            let (ir, inputs, norms) = defn.lower_grad_norms().map_err(TrainingError::CompilingBackwards)?;
            let func = compile(device, ir)?;

            let sqnorms = Buffer::zeroed(device, DType::F32, norms.names.len()).map_err(TrainingError::Unexpected)?;
            let mut tensors = BTreeMap::new();
            tensors.insert(norms.acc, sqnorms.clone());

            for (name, node) in inputs {
                tensors.insert(node, run.ranks[0].bound.grads[&name].clone());
            }

            run.norms = Some(NormRun { names: norms.names, sqnorms, func: Some(func), tensors });
        }

        self.run = Some(run);

        Ok(())
    }

    pub fn load_batch(&mut self, batch: &PreparedBatchHost<G>) -> Result<(), G::Error> {
        let run = self.run.as_ref().ok_or_else(|| NOT_PREPARED.to_string())?;

        let mut uploads = Upload::default();
        for (replica, rank) in self.replicas.iter().zip(&run.ranks) {
            uploads.extend(rank.bound.upload(&replica.copy, batch, rank.rows.start)?);
        }

        uploads.sync()
    }

    pub fn step(
        &mut self,
        lr: f32,
        next: &mut dyn FnMut() -> Option<PreparedBatchHost<G>>,
        pre_update: &mut [Box<dyn AdditionalUpdate<G>>],
        post_update: &mut [Box<dyn AdditionalUpdate<G>>],
    ) -> Result<StepOutput, TrainingError<G>> {
        if self.ranks() == 1 {
            self.step_single(lr, next, pre_update, post_update)
        } else {
            self.step_multi(lr, next, pre_update, post_update)
        }
    }

    fn step_single(
        &mut self,
        lr: f32,
        next: &mut dyn FnMut() -> Option<PreparedBatchHost<G>>,
        pre_update: &mut [Box<dyn AdditionalUpdate<G>>],
        post_update: &mut [Box<dyn AdditionalUpdate<G>>],
    ) -> Result<StepOutput, TrainingError<G>> {
        let Self { replicas, run, .. } = self;
        let run = run.as_mut().ok_or_else(not_prepared)?;
        let compute = replicas.iter().map(|replica| replica.compute.clone()).collect::<Vec<_>>();
        let copy = replicas.iter().map(|replica| replica.copy.clone()).collect::<Vec<_>>();
        let lr = TValue::F32(vec![lr]);

        let lrs = run.copy_lr(&compute, &lr)?;
        let backward = run.backward(&compute)?;

        let rank = &run.ranks[0];
        let (tgf, tlr) = (rank.tgf.clone(), rank.tlr.clone());
        let grads = &rank.bound.grads;
        let update = update_replica(&mut replicas[0], pre_update, post_update, &compute[0], tgf, tlr, grads);
        let update = update.map_err(TrainingError::OptimiserUpdateError)?;

        let host = next();
        let uploads = run.upload_next(&copy, host.as_ref())?;

        update.sync().map_err(TrainingError::Unexpected)?;
        sync_all(backward).and(sync_all(lrs)).and(uploads.sync()).map_err(TrainingError::Unexpected)?;

        if host.is_some() {
            swap_batches(run.ranks.iter_mut().map(|rank| &mut rank.bound));
        }

        run.output(&copy, host.is_some())
    }

    fn step_multi(
        &mut self,
        lr: f32,
        next: &mut dyn FnMut() -> Option<PreparedBatchHost<G>>,
        pre_update: &mut [Box<dyn AdditionalUpdate<G>>],
        post_update: &mut [Box<dyn AdditionalUpdate<G>>],
    ) -> Result<StepOutput, TrainingError<G>> {
        let Self { replicas, comm, run } = self;
        let run = run.as_mut().ok_or_else(not_prepared)?;
        let compute = replicas.iter().map(|replica| replica.compute.clone()).collect::<Vec<_>>();
        let copy = replicas.iter().map(|replica| replica.copy.clone()).collect::<Vec<_>>();
        let lr = TValue::F32(vec![lr]);

        let lrs = run.copy_lr(&compute, &lr)?;
        let backward = run.backward(&compute)?;

        let host = next();
        let uploads = run.upload_next(&copy, host.as_ref())?;

        unsafe { release_all(backward) };

        drain(reduce_and_update(comm, replicas, run, &compute, pre_update), &compute)?;
        sync_all(lrs).and(uploads.sync()).map_err(TrainingError::Unexpected)?;

        if host.is_some() {
            swap_batches(run.ranks.iter_mut().map(|rank| &mut rank.bound));
        }

        apply_hooks(post_update, replicas)?;

        run.output(&copy, host.is_some())
    }

    pub fn finish(&mut self) {
        self.run = None;
    }
}

impl<G: Gpu> RunState<G> {
    fn copy_lr<'a>(
        &self,
        compute: &[Arc<Stream<G>>],
        lr: &'a TValue,
    ) -> Result<Vec<SyncOnValue<G, &'a TValue>>, TrainingError<G>> {
        let lrs = self.ranks.iter().zip(compute).map(|(rank, compute)| rank.tlr.copy_from_host_async(compute, lr));
        lrs.collect::<Result<_, _>>().map_err(TrainingError::Unexpected)
    }

    fn backward(&self, compute: &[Arc<Stream<G>>]) -> Result<Vec<SyncOnValue<G, &Function<G>>>, TrainingError<G>> {
        let backward = self
            .ranks
            .iter()
            .zip(compute)
            .map(|(rank, compute)| rank.func.execute(compute.clone(), &rank.bound.tensors));
        backward.collect::<Result<_, _>>().map_err(TrainingError::GradientCalculationError)
    }

    fn upload_next<'a>(
        &'a self,
        copy: &[Arc<Stream<G>>],
        host: Option<&'a PreparedBatchHost<G>>,
    ) -> Result<Upload<'a, G>, TrainingError<G>> {
        let mut uploads = Upload::default();

        if let Some(host) = host {
            for (rank, copy) in self.ranks.iter().zip(copy) {
                let upload = rank.bound.upload_next(copy, host, rank.rows.start);
                uploads.extend(upload.map_err(TrainingError::Unexpected)?);
            }
        }

        Ok(uploads)
    }

    fn output(&self, copy: &[Arc<Stream<G>>], next_queued: bool) -> Result<StepOutput, TrainingError<G>> {
        let mut losses = Vec::new();
        for (rank, copy) in self.ranks.iter().zip(copy) {
            losses.push(rank.bound.loss.to_host_async(copy).map_err(TrainingError::Unexpected)?);
        }

        let mut loss = 0.0;
        for value in losses {
            loss += value.value().map_err(TrainingError::Unexpected)?.f32()[0];
        }

        let grad_sqnorms = match &self.norms {
            Some(norms) => {
                let sqnorms = norms.sqnorms.to_host_async(&copy[0]).and_then(SyncOnValue::value);
                let sqnorms = sqnorms.map_err(TrainingError::Unexpected)?;
                Some(norms.names.iter().cloned().zip(sqnorms.f32().iter().copied()).collect())
            }
            None => None,
        };

        Ok(StepOutput { loss, grad_sqnorms, next_queued })
    }
}

impl<G: Gpu> RankRun<G> {
    fn new<S>(
        defn: &ModelDefinition,
        replica: &Replica<G, S>,
        lowered: &ModelFunctionDefinition,
        gmap: &BTreeMap<NodeId, NodeId>,
        rows: Range<usize>,
        batch_size: usize,
    ) -> Result<Self, TrainingError<G>> {
        let device = &replica.device;
        let from_host = |value| Buffer::from_host(device, &TValue::F32(vec![value])).map_err(TrainingError::Unexpected);

        let func = compile(device, lowered.ir().clone())?;
        let bound = Bound::new(defn, lowered, gmap, &replica.weights, device, rows.len(), batch_size);

        Ok(Self {
            func,
            bound: bound.map_err(TrainingError::Unexpected)?,
            rows,
            tgf: from_host(1.0 / batch_size as f32)?,
            tlr: from_host(0.0)?,
        })
    }
}

impl<G: Gpu, S: OptimiserState<G>> Replica<G, S> {
    fn new(device: &Arc<Device<G>>, weights: TensorMap<G>, params: S::Params) -> Result<Self, G::Error> {
        let state =
            weights.iter().map(|(id, value)| S::new(device, value.size(), params.clone()).map(|s| (id.clone(), s)));
        let state = state.collect::<Result<_, _>>()?;
        let (compute, copy) = (device.new_stream()?, device.new_stream()?);

        Ok(Self { device: device.clone(), compute, copy, weights, state })
    }
}

impl<G: Gpu> Bound<G> {
    fn new(
        defn: &ModelDefinition,
        lowered: &ModelFunctionDefinition,
        gmap: &BTreeMap<NodeId, NodeId>,
        weights: &TensorMap<G>,
        device: &Arc<Device<G>>,
        batch_size: usize,
        host_batch: usize,
    ) -> Result<Self, G::Error> {
        let map = lowered.map();
        let mut tensors = BTreeMap::new();
        let mut grads = BTreeMap::new();

        for (mid, (name, _)) in defn.ir().weights() {
            let ty = defn.ir().node(*mid).ty();
            let Layout::Dense(dtype) = ty.layout() else { unreachable!() };
            let grad = Buffer::zeroed(device, dtype, ty.shape().size())?;

            tensors.insert(map[mid], weights[name].clone());
            tensors.insert(gmap[mid], grad.clone());
            grads.insert(name.clone(), grad);
        }

        let mut inputs = Vec::new();
        let mut batch = BTreeMap::new();
        let mut next = BTreeMap::new();
        let mut csr = BTreeMap::new();

        for (mid, name) in defn.ir().inputs() {
            let ty = defn.ir().node(*mid).ty();
            let tty = ty.ttype(batch_size);
            let stride = if ty.is_batched() { ty.ttype(1).size().get() } else { 0 };
            let host_size = ty.ttype(host_batch).size().get();

            if let Layout::Sparse(nnz) = ty.layout() {
                let offsets = Buffer::zeroed(device, DType::I32, batch_size + 1)?;
                let indices = Buffer::zeroed(device, DType::I32, batch_size * nnz)?;
                let expand = build_expand(batch_size, nnz, device.props()).unwrap().compile(device.clone())?;
                csr.insert(name.clone(), Csr { offsets, indices, expand });
            }

            let buf = Buffer::zeroed(device, tty.dtype(), tty.size().get())?;
            tensors.insert(map[mid], buf.clone());
            batch.insert(name.clone(), buf);
            next.insert(name.clone(), Buffer::zeroed(device, tty.dtype(), tty.size().get())?);
            inputs.push(RankInput { node: map[mid], name: name.clone(), stride, host_size });
        }

        let loss = Buffer::from_host(device, &TValue::F32(vec![0.0]))?;
        tensors.insert(map[&defn.loss().unwrap()], loss.clone());

        Ok(Self { tensors, grads, batch, next, loss, inputs, csr, host_batch })
    }

    fn upload<'a>(
        &'a self,
        stream: &Arc<Stream<G>>,
        host: &'a PreparedBatchHost<G>,
        start: usize,
    ) -> Result<Upload<'a, G>, G::Error> {
        self.upload_into(&self.batch, stream, host, start)
    }

    fn upload_next<'a>(
        &'a self,
        stream: &Arc<Stream<G>>,
        host: &'a PreparedBatchHost<G>,
        start: usize,
    ) -> Result<Upload<'a, G>, G::Error> {
        self.upload_into(&self.next, stream, host, start)
    }

    fn upload_into<'a>(
        &'a self,
        dst: &TensorMap<G>,
        stream: &Arc<Stream<G>>,
        host: &'a PreparedBatchHost<G>,
        start: usize,
    ) -> Result<Upload<'a, G>, G::Error> {
        let mut upload = Upload::default();

        for RankInput { name, stride, host_size, .. } in &self.inputs {
            if let Some(sparse) = host.sparse(name) {
                let csr = self.csr.get(name).ok_or_else(|| "Mismatched layout!".to_string())?;
                let rows = csr.offsets.size() - 1;

                if sparse.offsets().size() != self.host_batch + 1 {
                    return Err(
                        format!("Mismatched sizes: {} != {}", sparse.offsets().size(), self.host_batch + 1).into()
                    );
                }

                if start + rows > self.host_batch {
                    return Err("Copy range out of bounds!".to_string().into());
                }

                let offsets = sparse.offsets().as_slice::<i32>().unwrap();
                let (first, last) = (offsets[start] as usize, offsets[start + rows] as usize);

                upload.copies.push(csr.offsets.copy_from_pinned_range_async(
                    stream,
                    sparse.offsets(),
                    start,
                    rows + 1,
                )?);

                if last > first {
                    let indices =
                        csr.indices.copy_from_pinned_range_async(stream, sparse.indices(), first, last - first);
                    upload.copies.push(indices?);
                }

                let inputs = vec![csr.offsets.clone(), csr.indices.clone()];
                upload.expands.push(csr.expand.execute(stream.clone(), inputs, vec![dst[name].clone()])?);
                continue;
            }

            let value = host.inputs.get(name).ok_or_else(|| "Missing input!".to_string())?;

            if value.size() != *host_size {
                return Err(format!("Mismatched sizes: {} != {host_size}", value.size()).into());
            }

            upload.copies.push(dst[name].copy_from_pinned_range_async(
                stream,
                value,
                start * stride,
                dst[name].size(),
            )?);
        }

        Ok(upload)
    }
}

impl<G: Gpu> Default for Upload<'_, G> {
    fn default() -> Self {
        Self { copies: Vec::new(), expands: Vec::new() }
    }
}

impl<G: Gpu> Upload<'_, G> {
    fn extend(&mut self, other: Self) {
        self.copies.extend(other.copies);
        self.expands.extend(other.expands);
    }

    fn sync(self) -> Result<(), G::Error> {
        sync_all(self.expands).and(sync_all(self.copies))
    }
}

fn build_expand(rows: usize, nnz: usize, props: &DeviceProps) -> Result<KernelSrc, IRError> {
    let builder = PointwiseBuilder::new(rows * nnz);

    let offsets = builder.new_buffer(TType::new(rows + 1, DType::I32));
    let indices = builder.new_buffer(TType::new(rows * nnz, DType::I32));
    let out = builder.new_buffer(TType::new(rows * nnz, DType::I32));

    let tid = builder.tid();
    let (b, j) = tid.div_rem(nnz as i32);
    let start = offsets.read(b, 0) - offsets.read(0, 0);
    let count = offsets.read(b + 1, 0) - offsets.read(b, 0);
    out.write(tid, indices.conditional_read(start + j, (count - j).is_positive(), -1, 0));

    unsafe { builder.inner().lower("expand_sparse".to_string(), props) }
}

fn swap_batches<'a, G: Gpu>(bounds: impl Iterator<Item = &'a mut Bound<G>>) {
    for bound in bounds {
        mem::swap(&mut bound.batch, &mut bound.next);

        for RankInput { node, name, .. } in &bound.inputs {
            bound.tensors.insert(*node, bound.batch[name].clone());
        }
    }
}

fn compile<G: Gpu>(device: &Arc<Device<G>>, ir: TensorIR) -> Result<Function<G>, TrainingError<G>> {
    let mut func = Function::new(device.clone(), ir).map_err(TrainingError::CompilingBackwards)?;
    func.prealloc().map_err(TrainingError::Unexpected)?;
    Ok(func)
}

fn update_replica<'a, G: Gpu, S: OptimiserState<G>>(
    replica: &'a mut Replica<G, S>,
    pre_update: &'a mut [Box<dyn AdditionalUpdate<G>>],
    post_update: &'a mut [Box<dyn AdditionalUpdate<G>>],
    stream: &Arc<Stream<G>>,
    gradient_factor: Arc<Buffer<G>>,
    learning_rate: Arc<Buffer<G>>,
    gradients: &TensorMap<G>,
) -> OptimiserUpdateResult<'a, G> {
    let mut sync = OptimiserUpdateSync::default();

    for additional in pre_update {
        sync.extend_by(additional.apply_update(&replica.weights)?);
    }

    let state = &mut replica.state;
    sync.extend_by(update_weights(state, &replica.weights, stream, gradient_factor, learning_rate, gradients)?);

    for additional in post_update {
        sync.extend_by(additional.apply_update(&replica.weights)?);
    }

    Ok(sync)
}

fn update_weights<'a, G: Gpu, S: OptimiserState<G>>(
    state: &'a mut BTreeMap<String, S>,
    weights: &TensorMap<G>,
    stream: &Arc<Stream<G>>,
    gradient_factor: Arc<Buffer<G>>,
    learning_rate: Arc<Buffer<G>>,
    gradients: &TensorMap<G>,
) -> OptimiserUpdateResult<'a, G> {
    let mut sync = OptimiserUpdateSync::default();

    for (id, single) in state {
        let weight = weights.get(id).unwrap();

        if let Some(grads) = gradients.get(id) {
            sync.extend_by(single.update(
                stream,
                weight.clone(),
                grads.clone(),
                gradient_factor.clone(),
                learning_rate.clone(),
            )?);
        }
    }

    Ok(sync)
}

fn reduce_and_update<G: Gpu, S: OptimiserState<G>>(
    comm: &Comm<G>,
    replicas: &mut [Replica<G, S>],
    run: &RunState<G>,
    compute: &[Arc<Stream<G>>],
    pre_update: &mut [Box<dyn AdditionalUpdate<G>>],
) -> Result<(), TrainingError<G>> {
    let grads = run.ranks.iter().map(|rank| rank.bound.grads.values().cloned().collect()).collect::<Vec<_>>();
    comm.all_reduce(compute, &grads).map_err(TrainingError::Unexpected)?;

    let norms = match &run.norms {
        Some(NormRun { func: Some(func), tensors, .. }) => {
            Some(func.execute(compute[0].clone(), tensors).map_err(TrainingError::Unexpected)?)
        }
        _ => None,
    };

    apply_hooks(pre_update, replicas)?;

    let mut updates = OptimiserUpdateSync::default();
    for ((replica, rank), stream) in replicas.iter_mut().zip(&run.ranks).zip(compute) {
        let (tgf, tlr) = (rank.tgf.clone(), rank.tlr.clone());
        let update = update_weights(&mut replica.state, &replica.weights, stream, tgf, tlr, &rank.bound.grads);
        updates.extend_by(update.map_err(TrainingError::OptimiserUpdateError)?);
    }

    updates.sync().map_err(TrainingError::Unexpected)?;
    norms.map_or(Ok(()), |norms| norms.value().map(drop)).map_err(TrainingError::Unexpected)
}

fn apply_hooks<G: Gpu, S>(
    hooks: &mut [Box<dyn AdditionalUpdate<G>>],
    replicas: &[Replica<G, S>],
) -> Result<(), TrainingError<G>> {
    for hook in hooks {
        for replica in replicas {
            let update = hook.apply_update(&replica.weights).map_err(TrainingError::OptimiserUpdateError)?;
            update.sync().map_err(TrainingError::Unexpected)?;
        }
    }

    Ok(())
}

fn not_prepared<G: Gpu>() -> TrainingError<G> {
    TrainingError::Unexpected(NOT_PREPARED.to_string().into())
}

fn check_devices<G: Gpu>(devices: &[Arc<Device<G>>]) -> Result<(), G::Error> {
    let Some(primary) = devices.first() else {
        return Err("No devices given!".to_string().into());
    };

    for (i, device) in devices.iter().enumerate() {
        if devices[..i].iter().any(|other| other.ordinal() == device.ordinal()) {
            return Err(format!("GPU{} given more than once!", device.ordinal()).into());
        }
    }

    let kind = |device: &Device<G>| device.props().arch().unwrap_or(device.props().name()).to_string();

    for device in &devices[1..] {
        if kind(device) != kind(primary) {
            let (a, x) = (primary.ordinal(), kind(primary));
            let (b, y) = (device.ordinal(), kind(device));
            return Err(
                format!("Data-parallel training needs matching GPUs, but GPU{a} is {x} and GPU{b} is {y}!").into()
            );
        }
    }

    Ok(())
}

fn is_summed_over_batch(definition: &ModelDefinition) -> bool {
    let Some(loss) = definition.loss() else {
        return true;
    };

    let ir = definition.ir().inner();
    let Ok(op) = ir.parent_op(loss).and_then(|op| ir.op(op)) else {
        return false;
    };

    op.data()
        .downcast::<Reduce>()
        .is_some_and(|&Reduce(ty, dim, reduction)| ty.is_batched() && dim == Dim::Batch && reduction == Reduction::Sum)
}
