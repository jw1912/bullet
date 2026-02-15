pub mod model;
pub mod optimiser;
pub mod run;
pub mod runtime;

use optimiser::{Optimiser, OptimiserState};
use run::{
    dataloader::{DataLoader, DataLoadingError},
    schedule::{TrainingSchedule, TrainingSteps},
};
use runtime::Device;

use std::time::Instant;

#[derive(Debug)]
pub enum TrainerError<D: Device> {
    DataLoadingError(DataLoadingError),
    GradientCalculationError(D::Error),
    OptimiserUpdateError(D::Error),
    Unexpected(D::Error),
    IoError,
}

impl<D: Device> From<DataLoadingError> for TrainerError<D> {
    fn from(value: DataLoadingError) -> Self {
        Self::DataLoadingError(value)
    }
}

pub struct Trainer<D: Device, O: OptimiserState<D>, S> {
    pub optimiser: Optimiser<D, O>,
    pub state: S,
}

impl<D: Device, O: OptimiserState<D>, S> Trainer<D, O, S> {
    pub fn train_custom(
        &mut self,
        schedule: TrainingSchedule,
        dataloader: impl DataLoader,
        batch_callback: impl FnMut(&mut Self, usize, usize, f32),
        superbatch_callback: impl FnMut(&mut Self, usize),
    ) -> Result<(), TrainerError<D>> {
        run::train_custom(self, schedule, dataloader, batch_callback, superbatch_callback)
    }

    pub fn measure_max_cpu_throughput(
        &self,
        dataloader: impl DataLoader,
        steps: TrainingSteps,
    ) -> Result<(), TrainerError<D>> {
        let mut batch_no = 0;
        let mut superbatch = steps.start_superbatch;

        let t = Instant::now();
        let mut total = 0;

        dataloader.map_batches(steps.batch_size, |batch| {
            batch_no += 1;
            total += batch.batch_size;

            if batch_no % steps.batches_per_superbatch == 0 {
                batch_no = 0;
                superbatch += 1;

                println!("{:.0} datapoints / sec", total as f64 / t.elapsed().as_secs_f64());

                if superbatch > steps.end_superbatch {
                    return true;
                }
            }

            false
        })?;

        Ok(())
    }
}
