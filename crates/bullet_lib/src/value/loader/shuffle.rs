use std::sync::{
    Arc,
    mpsc::{self, SyncSender},
};

use bullet_trainer::reader::DataReader;
use rand_pcg::Pcg64Mcg;
use rayon::{ThreadPool, ThreadPoolBuilder};
use rip_shuffle::RipShuffleParallel;

use super::rng::seeded_rng;

#[derive(Clone)]
pub struct ShuffledDataLoader<R> {
    reader: R,
    buffer_size_mb: usize,
    threads: usize,
}

impl<R> ShuffledDataLoader<R> {
    pub fn new(reader: R, buffer_size_mb: usize, threads: usize) -> Self {
        Self { reader, buffer_size_mb, threads: threads.max(1) }
    }
}

impl<R, T> DataReader<T> for ShuffledDataLoader<R>
where
    R: DataReader<T>,
    T: Clone + Send + Sync,
{
    fn read_chunks<F: FnMut(&[T]) -> bool>(&self, skip_count: usize, mut f: F) {
        let buffer_size = (self.buffer_size_mb * 1024 * 1024 / size_of::<T>().max(1) / 2).max(1);

        std::thread::scope(|scope| {
            let (sender, receiver) = mpsc::sync_channel(0);
            let (stop_sender, stop_receiver) = mpsc::sync_channel::<()>(0);

            scope.spawn(move || {
                let mut shuffle = Shuffle::new(self.threads);
                let mut buffer = Vec::with_capacity(buffer_size);
                let mut stopped = false;

                self.reader.read_chunks(skip_count, |mut data| {
                    if matches!(stop_receiver.try_recv(), Err(mpsc::TryRecvError::Disconnected)) {
                        stopped = true;
                        return true;
                    }

                    if data.is_empty() {
                        stopped = sender.send(Vec::new()).is_err();
                        return stopped;
                    }

                    while !data.is_empty() {
                        let count = data.len().min(buffer_size - buffer.len());
                        buffer.extend_from_slice(&data[..count]);
                        data = &data[count..];

                        if buffer.len() == buffer_size {
                            shuffle.shuffle(&mut buffer);

                            if sender.send(std::mem::take(&mut buffer)).is_err() {
                                stopped = true;
                                return true;
                            }

                            buffer.reserve_exact(buffer_size);
                        }
                    }

                    false
                });

                if !stopped && !buffer.is_empty() {
                    shuffle.shuffle(&mut buffer);
                    let _ = sender.send(buffer);
                }
            });

            for buffer in receiver {
                if f(&buffer) {
                    break;
                }
            }

            drop(stop_sender);
        });
    }
}

pub(super) struct Shuffle {
    pub pool: Arc<ThreadPool>,
    rng: Pcg64Mcg,
}

impl Shuffle {
    pub fn new(threads: usize) -> Self {
        let mut rng = seeded_rng();
        let seed = (u128::from(rng.rand_u64()) << 64) | u128::from(rng.rand_u64());

        Self {
            pool: Arc::new(ThreadPoolBuilder::new().num_threads(threads.max(1)).build().unwrap()),
            rng: Pcg64Mcg::new(seed),
        }
    }

    pub fn shuffle<T: Send + Sync>(&mut self, data: &mut [T]) {
        self.pool.install(|| data.par_shuffle(&mut self.rng));
    }
}

pub(super) fn convert_chunks<T, U, F>(
    pool: &ThreadPool,
    sender: &SyncSender<Vec<U>>,
    data: &[T],
    chunk_size: usize,
    convert: F,
) -> bool
where
    T: Sync,
    U: Send,
    F: Fn(&[T]) -> Vec<U> + Sync,
{
    pool.in_place_scope(|scope| {
        let (chunk_sender, chunk_receiver) = mpsc::channel();
        let convert = &convert;
        let mut chunks = data.chunks(chunk_size.max(1));
        let mut pending = 0;

        loop {
            while pending < pool.current_num_threads() {
                let Some(chunk) = chunks.next() else { break };
                let chunk_sender = chunk_sender.clone();

                scope.spawn(move |_| {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| convert(chunk)));
                    let _ = chunk_sender.send(result);
                });

                pending += 1;
            }

            if pending == 0 {
                return false;
            }

            let buffer = match chunk_receiver.recv().unwrap() {
                Ok(buffer) => buffer,
                Err(error) => std::panic::resume_unwind(error),
            };
            pending -= 1;

            if sender.send(buffer).is_err() {
                return true;
            }
        }
    })
}
