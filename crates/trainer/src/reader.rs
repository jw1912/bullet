mod fixed_size;

pub use fixed_size::{FixedSizeData, FixedSizeDataReader};

use crate::{
    model::ModelInputsMapper,
    run::{DataLoader, DataLoadingError, PreparedBatchHost},
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
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        start_batch: usize,
        batch_size: usize,
        mut f: F,
    ) -> Result<(), DataLoadingError> {
        let mut batch = start_batch;
        let mut incomplete_buf = Vec::new();

        self.reader.read_chunks(start_batch * batch_size, |chunk| {
            let remainder = if !incomplete_buf.is_empty() {
                let remainder = batch_size - incomplete_buf.len();

                if chunk.len() >= remainder {
                    incomplete_buf.extend_from_slice(&chunk[..remainder]);
                    let prepared = self.mapper.map(&incomplete_buf, batch, self.threads);
                    batch += 1;

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
                    let prepared = self.mapper.map(data, batch, self.threads);
                    batch += 1;

                    if f(prepared) {
                        return true;
                    }
                }
            }

            false
        });

        Ok(())
    }
}
