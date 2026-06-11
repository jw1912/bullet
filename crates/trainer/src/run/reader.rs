use crate::{
    model::ModelInputsMapper,
    run::dataloader::{DataLoader, DataLoadingError, PreparedBatchHost},
};

pub trait DataReader<T> {
    fn read_chunks<F: FnMut(&[T]) -> bool>(&self, skip_count: usize, f: F);
}

pub struct ReadMapLoader<R, D> {
    reader: R,
    mapper: ModelInputsMapper<D>,
    start_batch: usize,
    threads: u8,
}

impl<R, D> DataLoader for ReadMapLoader<R, D>
where
    R: DataReader<D> + Send + Sync + 'static,
    D: Clone + Send + Sync + 'static,
{
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        batch_size: usize,
        mut f: F,
    ) -> Result<(), DataLoadingError> {
        let mut batch = self.start_batch;
        let mut incomplete_buf = Vec::new();

        self.reader.read_chunks(self.start_batch * batch_size, |chunk| {
            let remainder = if !incomplete_buf.is_empty() {
                let remainder = batch_size - incomplete_buf.len();

                if chunk.len() >= remainder {
                    incomplete_buf.extend_from_slice(&chunk[..remainder]);
                    let prepared = self.mapper.map(&incomplete_buf, batch, self.threads);
                    batch += 1;

                    if f(PreparedBatchHost { inputs: prepared }) {
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

                    if f(PreparedBatchHost { inputs: prepared }) {
                        return true;
                    }
                }
            }

            false
        });

        Ok(())
    }
}
