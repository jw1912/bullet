use bullet_compiler::tensor::TValue;

use crate::model::ModelInputsMapper;

pub trait DataReader<T> {
    fn read_chunks<F: FnMut(&[T]) -> bool>(&self, skip_count: usize, f: F);
}

pub fn map_batches<T: Clone, R>(
    reader: &impl DataReader<T>,
    mapper: &ModelInputsMapper<T>,
    start_batch: usize,
    batch_size: usize,
    threads: u8,
    mut f: impl FnMut(Vec<TValue>) -> bool,
) {
    let mut batch = start_batch;
    let mut incomplete_buf = Vec::new();

    reader.read_chunks(start_batch * batch_size, |chunk| {
        let remainder = if !incomplete_buf.is_empty() {
            let remainder = batch_size - incomplete_buf.len();

            if chunk.len() >= remainder {
                incomplete_buf.extend_from_slice(&chunk[..remainder]);
                let prepared = mapper.map(&incomplete_buf, batch, threads);
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
                let prepared = mapper.map(data, batch, threads);
                batch += 1;

                if f(prepared) {
                    return true;
                }
            }
        }

        false
    });
}
