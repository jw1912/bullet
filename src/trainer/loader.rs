use bulletformat::BulletFormat;

use crate::{
    inputs::InputType, lr::LrScheduler, outputs::OutputBuckets, tensor::Shape, wdl::WdlScheduler, DataLoader,
    TrainingSchedule,
};

use super::NetworkTrainer;

pub trait DataPreparer<T>: Send + Sync {
    type AdditionalArgs: Clone + Send + Sync + 'static;

    fn load(args: Self::AdditionalArgs, data: &[T], threads: usize, blend: f32, rscale: f32) -> Self;
}

pub fn create_dataloader<T, NT: NetworkTrainer, LD: DataLoader<T>, LR: LrScheduler, WDL: WdlScheduler>(
    schedule: TrainingSchedule<LR, WDL>,
    data_loader: &LD,
    batch_size: usize,
    threads: usize,
    sender: std::sync::mpsc::SyncSender<NT::PreparedData>,
    getters: <NT::PreparedData as DataPreparer<T>>::AdditionalArgs,
) -> std::thread::JoinHandle<()>
where
    NT::PreparedData: DataPreparer<T> + 'static,
{
    let rscale = 1.0 / schedule.eval_scale;

    let this_loader = data_loader.clone();

    std::thread::spawn(move || {
        let mut curr_superbatch = schedule.steps.start_superbatch;
        let mut curr_batch = 0;

        this_loader.map_batches(batch_size, |batch| {
            let blend = schedule.wdl_scheduler.blend(curr_batch, curr_superbatch, schedule.steps.end_superbatch);

            let prepared_data = NT::PreparedData::load(getters.clone(), batch, threads, blend, rscale);

            sender.send(prepared_data).unwrap();

            curr_batch += 1;

            let mut should_break = false;

            if curr_batch % schedule.steps.batches_per_superbatch == 0 {
                if curr_superbatch == schedule.steps.end_superbatch {
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
    stm: SparseInput,
    nstm: SparseInput,
    buckets: SparseInput,
    results: DenseInput,
}

impl<I: InputType, O: OutputBuckets<I::RequiredDataType>> DefaultDataPreparer<I, O> {
    pub fn stm(&self) -> &SparseInput {
        &self.stm
    }

    pub fn nstm(&self) -> &SparseInput {
        &self.nstm
    }

    pub fn buckets(&self) -> &SparseInput {
        &self.buckets
    }

    pub fn results(&self) -> &DenseInput {
        &self.results
    }
}

impl<I: InputType, O: OutputBuckets<I::RequiredDataType>> DataPreparer<I::RequiredDataType>
    for DefaultDataPreparer<I, O>
{
    type AdditionalArgs = (I, O);

    fn load(
        (input_getter, output_getter): (I, O),
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        rscale: f32,
    ) -> Self {
        let batch_size = data.len();
        let max_active = input_getter.max_active_inputs();
        let chunk_size = (batch_size + threads - 1) / threads;

        let input_size = input_getter.size();

        let mut prep = Self {
            input_getter,
            output_getter,
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
