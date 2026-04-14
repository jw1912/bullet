pub mod model;
pub mod optimiser;
pub mod run;

use bullet_gpu::runtime::Gpu;
use optimiser::{Optimiser, OptimiserState};
use run::{
    dataloader::{DataLoader, DataLoadingError},
    schedule::{TrainingSchedule, TrainingSteps},
};

use std::time::Instant;

#[derive(Debug)]
pub enum TrainerError<G: Gpu> {
    DataLoadingError(DataLoadingError),
    GradientCalculationError(G::Error),
    OptimiserUpdateError(G::Error),
    Unexpected(G::Error),
    IoError,
}

impl<G: Gpu> From<DataLoadingError> for TrainerError<G> {
    fn from(value: DataLoadingError) -> Self {
        Self::DataLoadingError(value)
    }
}

pub struct Trainer<G: Gpu, O: OptimiserState<G>, S> {
    pub optimiser: Optimiser<G, O>,
    pub state: S,
}

impl<G: Gpu, O: OptimiserState<G>, S> Trainer<G, O, S> {
    pub fn train_custom(
        &mut self,
        schedule: TrainingSchedule,
        dataloader: impl DataLoader,
        batch_callback: impl FnMut(&mut Self, usize, usize, f32),
        superbatch_callback: impl FnMut(&mut Self, usize),
    ) -> Result<(), TrainerError<G>> {
        run::train_custom(self, schedule, dataloader, batch_callback, superbatch_callback)
    }

    pub fn measure_max_cpu_throughput(
        &self,
        dataloader: impl DataLoader,
        steps: TrainingSteps,
    ) -> Result<(), TrainerError<G>> {
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
