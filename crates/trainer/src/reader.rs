mod fixed_size;

pub use fixed_size::{FixedSizeData, FixedSizeDataReader};

use std::sync::Arc;

use bullet_gpu::runtime::Gpu;

use crate::{
    model::ModelInputsMapper,
    run::{DataLoader, DataLoadingError, HostPool, PreparedBatchHost, Step},
};

pub trait DataReader<T>: Clone + Send + Sync + 'static {
    fn read_chunks<F: FnMut(&[T]) -> bool>(&self, skip_count: usize, f: F);
}

pub struct ReadMapLoader<R, D> {
    reader: R,
    mapper: ModelInputsMapper<D>,
    threads: u8,
}

impl<R, D> ReadMapLoader<R, D> {
    pub fn new(reader: R, mapper: ModelInputsMapper<D>, threads: u8) -> Self {
        Self { reader, mapper, threads }
    }
}

impl<R, D> DataLoader for ReadMapLoader<R, D>
where
    R: DataReader<D>,
    D: Clone + Send + Sync + 'static,
{
    fn map_batches<G: Gpu, F: FnMut(PreparedBatchHost<G>) -> bool>(
        self,
        pool: &Arc<HostPool<G>>,
        start: Step,
        batch_size: usize,
        mut f: F,
    ) -> Result<(), DataLoadingError> {
        let mut step = start;
        let mut incomplete_buf = Vec::new();
        let mut error = None;

        let mut map = |data: &[D], step: Step| {
            let prepared = self.mapper.map(pool, data, step, self.threads);
            prepared.map_err(|e| error = Some(DataLoadingError::Message(format!("{e:?}")))).ok()
        };

        self.reader.read_chunks(batch_size * start.total_batches(), |chunk| {
            let remainder = if !incomplete_buf.is_empty() {
                let remainder = batch_size - incomplete_buf.len();

                if chunk.len() >= remainder {
                    incomplete_buf.extend_from_slice(&chunk[..remainder]);
                    let Some(prepared) = map(&incomplete_buf, step) else { return true };
                    step.step();

                    if f(prepared) {
                        return true;
                    }

                    incomplete_buf.clear();
                } else {
                    incomplete_buf.extend_from_slice(chunk);
                }

                remainder
            } else {
                0
            };

            if chunk.len() >= remainder {
                let chunks = chunk[remainder..chunk.len()].chunks_exact(batch_size);
                incomplete_buf.extend_from_slice(chunks.remainder());

                for data in chunks {
                    let Some(prepared) = map(data, step) else { return true };
                    step.step();

                    if f(prepared) {
                        return true;
                    }
                }
            }

            false
        });

        error.map_or(Ok(()), Err)
    }
}
