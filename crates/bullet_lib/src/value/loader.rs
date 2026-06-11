mod direct;
mod montybinpack;
mod rng;
pub mod sfbinpack;
mod text;
pub mod viribinpack;

use bullet_compiler::tensor::TValue;
use bullet_trainer::run::dataloader::PreparedBatchHost;
pub use direct::{CanBeDirectlySequentiallyLoaded, DirectSequentialDataLoader};
pub use montybinpack::MontyBinpackLoader;
pub use sfbinpack::SfBinpackLoader;
pub use text::InMemoryTextLoader;
pub use viribinpack::{ViriBinpackLoader, ViriFilter};

use bulletformat::BulletFormat;

use crate::game::{inputs::SparseInputType, outputs::OutputBuckets};

use super::Wgt;

unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::ChessBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::AtaxxBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::CudADFormat {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::MarlinFormat {}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

pub trait LoadableDataType: Sized {
    fn score(&self) -> i16;

    fn result(&self) -> GameResult;
}

impl<T: BulletFormat + 'static> LoadableDataType for T {
    fn result(&self) -> GameResult {
        [GameResult::Loss, GameResult::Draw, GameResult::Win][self.result_idx()]
    }

    fn score(&self) -> i16 {
        <Self as BulletFormat>::score(self)
    }
}

/// Dictates how data is read from a file into the expected datatype.
/// This allows for the file format to be divorced from the training
/// data format.
pub trait DataLoader<T>: Clone + Send + Sync + 'static {
    fn data_file_paths(&self) -> &[String];

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_chunks<F: FnMut(&[T]) -> bool>(&self, start_position: usize, f: F);
}

pub(crate) type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;

#[derive(Clone)]
pub struct DefaultDataLoader<I: SparseInputType, O, D> {
    input_getter: I,
    output_getter: O,
    blend_getter: B<I>,
    weight_getter: Option<Wgt<I>>,
    use_win_rate_model: bool,
    wdl: bool,
    scale: f32,
    loader: D,
}

impl<I: SparseInputType, O, D> DefaultDataLoader<I, O, D> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        scale: f32,
        loader: D,
    ) -> Self {
        Self { input_getter, output_getter, blend_getter, weight_getter, use_win_rate_model, wdl, scale, loader }
    }
}

impl<I, O, D> DefaultDataLoader<I, O, D>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    D: DataLoader<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    pub fn load_and_map_batches<F: FnMut(&[I::RequiredDataType]) -> bool>(
        &self,
        start_batch: usize,
        batch_size: usize,
        mut f: F,
    ) {
        let mut incomplete_buf = Vec::new();

        self.loader.map_chunks(start_batch * batch_size, |chunk| {
            let remainder = if !incomplete_buf.is_empty() {
                let remainder = batch_size - incomplete_buf.len();

                if chunk.len() >= remainder {
                    incomplete_buf.extend_from_slice(&chunk[..remainder]);
                    let should_break = f(&incomplete_buf);
                    incomplete_buf.clear();

                    if should_break {
                        return true;
                    }
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

                for batch in chunks {
                    let should_break = f(batch);

                    if should_break {
                        return true;
                    }
                }
            }

            false
        });
    }

    pub fn prepare(&self, data: &[I::RequiredDataType], threads: usize, blend: f32) -> PreparedBatchHost {
        prepare(
            &self.input_getter,
            &self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            data,
            threads,
            blend,
            self.scale,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare<I, O>(
    input_getter: &I,
    output_getter: &O,
    blend_getter: B<I>,
    weight_getter: Option<Wgt<I>>,
    use_win_rate_model: bool,
    wdl: bool,
    data: &[I::RequiredDataType],
    threads: usize,
    blend: f32,
    scale: f32,
) -> PreparedBatchHost
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    let rscale = 1.0 / scale;
    let batch_size = data.len();
    let max_active = input_getter.max_active();
    let chunk_size = batch_size.div_ceil(threads);
    let input_size = input_getter.num_inputs();
    let output_size = if wdl { 3 } else { 1 };
    let sparse_size = max_active * batch_size;

    let mut stm = vec![0; sparse_size];
    let mut nstm = vec![0; sparse_size];
    let mut buckets = vec![0; batch_size];
    let mut targets = vec![0.0; output_size * batch_size];
    let mut weights = vec![0.0; batch_size];

    let sparse_chunk_size = max_active * chunk_size;

    std::thread::scope(|s| {
        data.chunks(chunk_size)
            .zip(stm.chunks_mut(sparse_chunk_size))
            .zip(nstm.chunks_mut(sparse_chunk_size))
            .zip(buckets.chunks_mut(chunk_size))
            .zip(targets.chunks_mut(output_size * chunk_size))
            .zip(weights.chunks_mut(chunk_size))
            .for_each(|(((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), results_chunk), weights_chunk)| {
                s.spawn(move || {
                    let chunk_len = data_chunk.len();

                    for i in 0..chunk_len {
                        let pos = &data_chunk[i];
                        let mut j = 0;
                        let sparse_offset = max_active * i;

                        input_getter.map_features(pos, |our, opp| {
                            assert!(our < input_size && opp < input_size, "Input feature index exceeded input size!");

                            stm_chunk[sparse_offset + j] = our as i32;
                            nstm_chunk[sparse_offset + j] = opp as i32;

                            j += 1;
                        });

                        for j in j..max_active {
                            stm_chunk[sparse_offset + j] = -1;
                            nstm_chunk[sparse_offset + j] = -1;
                        }

                        assert!(j <= max_active, "More inputs provided than the specified maximum!");

                        buckets_chunk[i] = i32::from(output_getter.bucket(pos));
                        weights_chunk[i] = weight_getter.map_or(1.0, |w| w(pos));

                        if wdl {
                            results_chunk[output_size * i + usize::from(pos.result() as u8)] = 1.0;
                        } else {
                            let score = f32::from(pos.score());
                            let score = if use_win_rate_model {
                                let p = (score - 270.0) / 380.0;
                                let pm = (-score - 270.0) / 380.0;
                                0.5 * (1.0 + sigmoid(p) - sigmoid(pm))
                            } else {
                                sigmoid(rscale * score)
                            };
                            let result = f32::from(pos.result() as u8) / 2.0;
                            let blend = blend_getter(pos, blend);
                            assert!((0.0..=1.0).contains(&blend), "WDL proportion must be in [0, 1]");
                            results_chunk[i] = blend * result + (1. - blend) * score;
                        }
                    }
                });
            });
    });

    PreparedBatchHost {
        inputs: [
            ("stm".to_string(), TValue::I32(stm)),
            ("nstm".to_string(), TValue::I32(nstm)),
            ("buckets".to_string(), TValue::I32(buckets)),
            ("targets".to_string(), TValue::F32(targets)),
            ("entry_weights".to_string(), TValue::F32(weights)),
        ]
        .into(),
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
