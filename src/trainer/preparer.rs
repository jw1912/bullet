use std::sync::mpsc::SyncSender;

use super::schedule::{wdl::WdlScheduler, TrainingSteps};

pub trait DataPreparer: Clone + Send + Sync {
    type DataType: Send + Sync;
    type PreparedData: Send + Sync;

    fn get_data_file_paths(&self) -> &[String];

    fn try_count_positions(&self) -> Option<u64> {
        None
    }

    fn load_and_map_batches<F: FnMut(&[Self::DataType]) -> bool>(&self, start_batch: usize, batch_size: usize, f: F);

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

        let start_batch = steps.batches_per_superbatch * (steps.start_superbatch - 1);

        preparer.load_and_map_batches(start_batch, steps.batch_size, |batch| {
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
