use std::{
    collections::{BTreeMap, HashMap},
    mem,
    sync::{Arc, Mutex, mpsc},
    thread,
};

use bullet_compiler::tensor::DType;
use bullet_gpu::{
    buffer::{Buffer, PinnedBuffer, SyncOnValue},
    runtime::{Device, Gpu, Stream},
};

use crate::{model::TensorMap, run::Step};

#[derive(Debug)]
pub enum DataLoadingError {
    TooManyBatchesReceived,
    NoBatchesReceived,
    Message(String),
}

pub trait DataLoader: Send + Sync + 'static {
    fn map_batches<G: Gpu, F: FnMut(PreparedBatchHost<G>) -> bool>(
        self,
        pool: &Arc<HostPool<G>>,
        start: Step,
        batch_size: usize,
        f: F,
    ) -> Result<(), DataLoadingError>;
}

pub struct PreparedBatchHost<G: Gpu> {
    pub inputs: BTreeMap<String, PinnedBuffer<G>>,
    sparse: BTreeMap<String, SparseBatch<G>>,
    pool: Arc<HostPool<G>>,
}

pub(crate) struct SparseBatch<G: Gpu> {
    offsets: PinnedBuffer<G>,
    indices: PinnedBuffer<G>,
}

impl<G: Gpu> PreparedBatchHost<G> {
    pub fn new(pool: Arc<HostPool<G>>, inputs: BTreeMap<String, PinnedBuffer<G>>) -> Self {
        Self { inputs, sparse: BTreeMap::new(), pool }
    }

    pub fn copy_to_device_async<'a>(
        &'a self,
        stream: &Arc<Stream<G>>,
        tensors: &TensorMap<G>,
    ) -> Result<Vec<SyncOnValue<G, &'a PinnedBuffer<G>>>, G::Error> {
        let mut syncs = Vec::new();

        for (id, tensor) in tensors {
            let value = self.inputs.get(id).ok_or_else(|| "Missing input!".into())?;
            syncs.push(tensor.copy_from_pinned_async(stream, value)?);
        }

        Ok(syncs)
    }

    pub(crate) fn sparse(&self, name: &str) -> Option<&SparseBatch<G>> {
        self.sparse.get(name)
    }

    pub(crate) fn compact(&mut self, sparse: &[(String, usize)], threads: usize) -> Result<(), G::Error> {
        let padded = sparse.iter().filter_map(|(name, nnz)| Some((name.clone(), *nnz, self.inputs.remove(name)?)));
        let padded = padded.collect::<Vec<_>>();

        let Some((_, nnz, value)) = padded.first() else { return Ok(()) };
        let rows = value.size() / nnz;

        if padded.iter().any(|(_, _, value)| value.dtype() != DType::I32) {
            return Err("Sparse inputs must be I32!".to_string().into());
        }

        if padded.iter().any(|(_, nnz, value)| value.size() != rows * nnz) {
            return Err("Mismatched sizes!".to_string().into());
        }

        let mut csr = Vec::new();
        for (_, nnz, _) in &padded {
            csr.push((self.pool.take(DType::I32, rows + 1)?, self.pool.take(DType::I32, rows * nnz)?));
        }

        let inputs = padded.iter().map(|(_, nnz, value)| (value.as_slice().unwrap(), *nnz)).collect::<Vec<_>>();
        let slices = csr.iter_mut().map(|(offsets, indices)| (offsets.as_mut_slice(), indices.as_mut_slice()));
        let (offsets, indices) = slices.map(|(offsets, indices)| (offsets.unwrap(), indices.unwrap())).unzip();
        let totals = compact_rows(&inputs, offsets, indices, rows, threads);

        for (((name, _, value), (offsets, mut indices)), total) in padded.into_iter().zip(csr).zip(totals) {
            indices.set_size(total)?;
            self.pool.give(value);
            self.sparse.insert(name, SparseBatch { offsets, indices });
        }

        Ok(())
    }

    pub fn to_device(self, device: &Arc<Device<G>>) -> Result<TensorMap<G>, G::Error> {
        self.inputs
            .iter()
            .map(|(id, value)| Buffer::from_pinned(device, value).map(|tensor| (id.clone(), tensor)))
            .collect()
    }
}

impl<G: Gpu> Drop for PreparedBatchHost<G> {
    fn drop(&mut self) {
        for (_, value) in mem::take(&mut self.inputs) {
            self.pool.give(value);
        }

        for (_, sparse) in mem::take(&mut self.sparse) {
            self.pool.give(sparse.offsets);
            self.pool.give(sparse.indices);
        }
    }
}

impl<G: Gpu> SparseBatch<G> {
    pub(crate) fn offsets(&self) -> &PinnedBuffer<G> {
        &self.offsets
    }

    pub(crate) fn indices(&self) -> &PinnedBuffer<G> {
        &self.indices
    }
}

type Chunk<'a> = (&'a [i32], usize, &'a mut [i32]);

fn compact_rows(
    inputs: &[(&[i32], usize)],
    offsets: Vec<&mut [i32]>,
    indices: Vec<&mut [i32]>,
    rows: usize,
    threads: usize,
) -> Vec<usize> {
    let chunk = rows.div_ceil(threads.max(1)).max(1);
    let mut work = (0..rows.div_ceil(chunk)).map(|_| Vec::<Chunk>::new()).collect::<Vec<_>>();

    for (&(padded, nnz), offsets) in inputs.iter().zip(offsets) {
        offsets[0] = 0;

        for (work, (padded, ends)) in
            work.iter_mut().zip(padded.chunks(chunk * nnz).zip(offsets[1..].chunks_mut(chunk)))
        {
            work.push((padded, nnz, ends));
        }
    }

    thread::scope(|s| {
        let (counted, counts) = mpsc::channel();
        let mut parts = Vec::new();

        for (id, mut work) in work.into_iter().enumerate() {
            let (send, recv) = mpsc::channel::<Vec<(i32, &mut [i32])>>();
            let counted = counted.clone();
            parts.push(send);

            s.spawn(move || {
                let totals = work.iter_mut().map(|(padded, nnz, ends)| {
                    let mut total = 0;

                    for (end, row) in ends.iter_mut().zip(padded.chunks(*nnz)) {
                        total += row.iter().position(|&x| x == -1).unwrap_or(*nnz) as i32;
                        *end = total;
                    }

                    total
                });

                counted.send((id, totals.collect::<Vec<_>>())).unwrap();
                drop(counted);

                for ((padded, nnz, ends), (base, part)) in work.into_iter().zip(recv.recv().unwrap()) {
                    let mut start = 0;

                    for (end, row) in ends.iter_mut().zip(padded.chunks(nnz)) {
                        let len = *end as usize - start;
                        part[start..start + len].copy_from_slice(&row[..len]);
                        start += len;
                        *end += base;
                    }
                }
            });
        }

        drop(counted);
        let mut counts = counts.iter().collect::<Vec<_>>();
        counts.sort_by_key(|(id, _)| *id);

        let mut splits = counts.iter().map(|_| Vec::new()).collect::<Vec<_>>();
        let mut totals = Vec::new();

        for (input, mut rest) in indices.into_iter().enumerate() {
            let mut base = 0;

            for (split, (_, count)) in splits.iter_mut().zip(&counts) {
                let (part, tail) = mem::take(&mut rest).split_at_mut(count[input] as usize);
                split.push((base, part));
                rest = tail;
                base += count[input];
            }

            totals.push(base as usize);
        }

        for (send, split) in parts.into_iter().zip(splits) {
            send.send(split).unwrap();
        }

        totals
    })
}

type FreeList<G> = HashMap<(DType, usize), Vec<PinnedBuffer<G>>>;

/// Pool of pinned host buffers for preparing batches in, as pinned
/// allocations are expensive and batches are generally the same size
pub struct HostPool<G: Gpu> {
    device: Arc<Device<G>>,
    free: Mutex<FreeList<G>>,
}

impl<G: Gpu> HostPool<G> {
    pub fn new(device: Arc<Device<G>>) -> Arc<Self> {
        Arc::new(Self { device, free: Mutex::default() })
    }

    pub fn device(&self) -> Arc<Device<G>> {
        self.device.clone()
    }

    /// Take a buffer of the given dtype and size from the pool, allocating
    /// a new one if needed. Contents of reused buffers are unspecified.
    pub fn take(&self, dtype: DType, size: usize) -> Result<PinnedBuffer<G>, G::Error> {
        let capacity = size.next_power_of_two();
        let cached = self.free.lock().unwrap().get_mut(&(dtype, capacity)).and_then(Vec::pop);

        let mut value = match cached {
            Some(value) => value,
            None => PinnedBuffer::zeroed(&self.device, dtype, capacity)?,
        };

        value.set_size(size)?;
        Ok(value)
    }

    fn give(&self, value: PinnedBuffer<G>) {
        self.free.lock().unwrap().entry((value.dtype(), value.capacity())).or_default().push(value);
    }
}
