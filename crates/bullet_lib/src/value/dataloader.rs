use bullet_compiler::ir::graph::TValue;
use bullet_trainer::run::{
    dataloader::{DataLoader, DataLoadingError, PreparedBatchHost},
    schedule::TrainingSteps,
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    trainer::schedule::wdl::WdlScheduler,
    value::loader::{self, PreparedData},
};

pub struct ValueDataLoader<I, O, D, W>
where
    I: SparseInputType,
    I::RequiredDataType: loader::LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
    D: loader::DataLoader<I::RequiredDataType>,
{
    pub dataloader: loader::DefaultDataLoader<I, O, D>,
    pub steps: TrainingSteps,
    pub threads: usize,
    pub wdl: W,
}

impl<I, O, D, W> DataLoader for ValueDataLoader<I, O, D, W>
where
    I: SparseInputType,
    I::RequiredDataType: loader::LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
    W: WdlScheduler,
    D: loader::DataLoader<I::RequiredDataType>,
{
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        batch_size: usize,
        mut f: F,
    ) -> Result<(), DataLoadingError> {
        let ValueDataLoader { dataloader, steps, threads, wdl } = self;
        let start_batch = steps.batches_per_superbatch * (steps.start_superbatch - 1);

        assert_eq!(batch_size, steps.batch_size);

        let mut batch_no = 0;
        let mut superbatch = 1;

        dataloader.load_and_map_batches(start_batch, batch_size, |batch| {
            let blend = wdl.blend(batch_no, superbatch, steps.end_superbatch);
            let prepared_data = dataloader.prepare(batch, threads, blend);

            batch_no += 1;

            if batch_no % steps.batches_per_superbatch == 0 {
                batch_no = 0;
                superbatch += 1;
            }

            f(prepared_data.into())
        });

        Ok(())
    }
}

impl<I: SparseInputType, O> From<PreparedData<I, O>> for PreparedBatchHost {
    fn from(prepared_data: PreparedData<I, O>) -> Self {
        PreparedBatchHost {
            batch_size: prepared_data.batch_size,
            inputs: [
                ("stm".to_string(), TValue::I32(prepared_data.stm.value)),
                ("nstm".to_string(), TValue::I32(prepared_data.nstm.value)),
                ("buckets".to_string(), TValue::I32(prepared_data.buckets.value)),
                ("targets".to_string(), TValue::F32(prepared_data.targets.value)),
                ("entry_weights".to_string(), TValue::F32(prepared_data.weights.value)),
            ]
            .into(),
        }
    }
}
