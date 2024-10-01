use std::sync::mpsc::SyncSender;

use bulletformat::BulletFormat;

use crate::{
    inputs::InputType, outputs::OutputBuckets, tensor::Shape,
};

use super::schedule::{TrainingSteps, wdl::WdlScheduler};

pub trait DataPreparer: Clone + Send + Sync {
    type DataType: Send + Sync;
    type PreparedData: Send + Sync;

    fn get_data_file_paths(&self) -> &[String];

    fn try_count_positions(&self) -> Option<u64> {
        None
    }

    fn load_and_map_batches<F: FnMut(&[Self::DataType]) -> bool>(&self, batch_size: usize, f: F);

    fn prepare(&self, data: &[Self::DataType], threads: usize, blend: f32) -> Self::PreparedData;
}

pub fn create_dataloader<D: DataPreparer + 'static, WDL: WdlScheduler>(
    preparer: D,
    sender: SyncSender<D::PreparedData>,
    steps: TrainingSteps,
    wdl: WDL,
    threads: usize,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut curr_superbatch = steps.start_superbatch;
        let mut curr_batch = 0;

        preparer.load_and_map_batches(steps.batch_size, |batch| {
            let blend = wdl.blend(curr_batch, curr_superbatch, steps.end_superbatch);

            let prepared_data = preparer.prepare(batch, threads, blend);

            sender.send(prepared_data).unwrap();

            curr_batch += 1;

            let mut should_break = false;

            if curr_batch % steps.batches_per_superbatch == 0 {
                if curr_superbatch == steps.end_superbatch {
                    should_break = true;
                }

                curr_batch = 0;
                curr_superbatch += 1;
            }

            should_break
        });
    })
}

pub struct DenseInput {
    pub shape: Shape,
    pub value: Vec<f32>,
}

#[derive(Clone)]
pub struct SparseInput {
    pub shape: Shape,
    pub value: Vec<i32>,
    pub max_active: usize,
}

impl Default for SparseInput {
    fn default() -> Self {
        Self { shape: Shape::new(0, 0), value: Vec::new(), max_active: 0 }
    }
}

/// A batch of data, in the correct format for the GPU.
pub struct DefaultDataPreparer<I, O> {
    input_getter: I,
    output_getter: O,
    pub batch_size: usize,
    pub stm: SparseInput,
    pub nstm: SparseInput,
    pub buckets: SparseInput,
    pub results: DenseInput,
}

impl<I: InputType, O: OutputBuckets<I::RequiredDataType>> DefaultDataPreparer<I, O> {
    pub fn prepare(
        input_getter: I,
        output_getter: O,
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
    ) -> Self {
        let rscale = 1.0 / scale;
        let batch_size = data.len();
        let max_active = input_getter.max_active_inputs();
        let chunk_size = (batch_size + threads - 1) / threads;

        let input_size = input_getter.size();

        let mut prep = Self {
            input_getter,
            output_getter,
            batch_size,
            stm: SparseInput {
                shape: Shape::new(input_size, batch_size),
                max_active,
                value: vec![0; max_active * batch_size],
            },
            nstm: SparseInput {
                shape: Shape::new(input_size, batch_size),
                max_active,
                value: vec![0; max_active * batch_size],
            },
            buckets: SparseInput {
                shape: Shape::new(input_size, batch_size),
                max_active: 1,
                value: vec![0; batch_size],
            },
            results: DenseInput { shape: Shape::new(1, batch_size), value: vec![0.0; batch_size] },
        };

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(prep.stm.value.chunks_mut(max_active * chunk_size))
                .zip(prep.nstm.value.chunks_mut(max_active * chunk_size))
                .zip(prep.buckets.value.chunks_mut(chunk_size))
                .zip(prep.results.value.chunks_mut(chunk_size))
                .for_each(|((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), results_chunk)| {
                    let inp = &prep.input_getter;
                    let out = &prep.output_getter;
                    s.spawn(move || {
                        let chunk_len = data_chunk.len();

                        for i in 0..chunk_len {
                            let pos = &data_chunk[i];
                            let mut j = 0;
                            let offset = max_active * i;

                            for (our, opp) in inp.feature_iter(pos) {
                                stm_chunk[offset + j] = our as i32;
                                nstm_chunk[offset + j] = opp as i32;
                                j += 1;
                            }

                            if j < max_active {
                                stm_chunk[offset + j] = -1;
                                nstm_chunk[offset + j] = -1;
                            }

                            assert!(j <= max_active, "More inputs provided than the specified maximum!");

                            results_chunk[i] = pos.blended_result(blend, rscale);
                            buckets_chunk[i] = i32::from(out.bucket(pos));
                        }
                    });
                });
        });

        prep
    }
}
