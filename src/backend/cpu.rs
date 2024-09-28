pub mod ops;
pub mod util;

#[derive(Clone, Copy)]
pub struct DeviceHandles {
    pub(crate) threads: usize,
}

impl Default for DeviceHandles {
    fn default() -> Self {
        Self { threads: 1 }
    }
}

impl DeviceHandles {
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads;
    }

    pub(crate) fn workload_chunks<F: Fn(usize, usize, usize) + Copy + Send>(&self, size: usize, workload_chunk: F) {
        let threads = self.threads;
        let chunk_size = (size + threads - 1) / threads;

        let mut covered = 0;

        std::thread::scope(|s| {
            for thread in 0..threads {
                let this_chunk_size = if covered + chunk_size > size { size - covered } else { chunk_size };

                let start_idx = covered;
                covered += this_chunk_size;
                assert!(covered <= size);

                s.spawn(move || {
                    workload_chunk(thread, start_idx, this_chunk_size);
                });
            }
        });
    }

    pub(crate) fn split_workload<F: Fn(usize, usize) + Copy + Send + Sync>(&self, size: usize, single_workload: F) {
        self.workload_chunks(size, |thread, start_idx, this_chunk_size| {
            for idx in start_idx..start_idx + this_chunk_size {
                single_workload(thread, idx);
            }
        });
    }
}
