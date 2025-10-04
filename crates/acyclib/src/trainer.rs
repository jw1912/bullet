pub mod dataloader;
pub mod logger;
pub mod optimiser;
pub mod schedule;

use dataloader::{DataLoader, PreparedBatchDevice, PreparedBatchHost};
use optimiser::{Optimiser, OptimiserState};
use schedule::TrainingSchedule;

use std::{sync::mpsc, thread, time::Instant};

use crate::{
    device::{Device, OperationError},
    graph::like::GraphLike,
};

#[derive(Debug)]
pub enum DataLoadingError {
    TooManyBatchesReceived,
    NoBatchesReceived,
}

#[derive(Debug)]
pub enum TrainerError<D: Device> {
    DataLoadingError(DataLoadingError),
    GradientCalculationError(OperationError<D::DeviceError>),
    Unexpected(OperationError<D::DeviceError>),
    MoreDevicesThanBatchSize(usize, usize),
    IoError,
}

impl<D: Device> From<DataLoadingError> for TrainerError<D> {
    fn from(value: DataLoadingError) -> Self {
        Self::DataLoadingError(value)
    }
}

pub struct Trainer<D: Device, G: GraphLike<D>, O: OptimiserState<D>, S> {
    pub optimiser: Optimiser<D, G, O>,
    pub state: S,
}

impl<D: Device, G: GraphLike<D>, O: OptimiserState<D>, S> Trainer<D, G, O, S> {
    pub fn train_custom(
        &mut self,
        schedule: TrainingSchedule,
        dataloader: impl DataLoader<Error = DataLoadingError>,
        mut batch_callback: impl FnMut(&mut Self, usize, usize, f32),
        mut superbatch_callback: impl FnMut(&mut Self, usize),
    ) -> Result<(), TrainerError<D>> {
        logger::clear_colours();
        println!("{}", logger::ansi("Beginning Training", "34;1"));

        let timer = Instant::now();
        let lr = schedule.lr_schedule;
        let steps = schedule.steps;

        if self.optimiser.graph.devices().len() > steps.batch_size {
            return Err(TrainerError::MoreDevicesThanBatchSize(self.optimiser.graph.devices().len(), steps.batch_size));
        }

        let (sender, receiver) = mpsc::sync_channel::<PreparedBatchHost>(32);

        let dataloader = thread::spawn(move || {
            let mut batch_no = 0;
            let mut superbatch = steps.start_superbatch;

            dataloader.map_batches(steps.batch_size, |batch| {
                sender.send(batch).unwrap();

                batch_no += 1;

                if batch_no % steps.batches_per_superbatch == 0 {
                    batch_no = 0;
                    superbatch += 1;

                    if superbatch > steps.end_superbatch {
                        return true;
                    }
                }

                false
            })
        });

        let mut prev_lr = lr(0, 1);
        let mut superbatch = steps.start_superbatch;
        let mut curr_batch = 0;
        let mut superbatch_timer = Instant::now();
        let mut running_loss = 0.0;
        let mut superbatch_positions = 0;

        let first_batch =
            receiver.recv().map_err(|_| TrainerError::DataLoadingError(DataLoadingError::NoBatchesReceived))?;

        let mut batch_on_device = PreparedBatchDevice::new(self.optimiser.graph.devices(), &first_batch)
            .map_err(|e| TrainerError::Unexpected(e.into()))?;

        let mut batch_queued = true;

        while batch_queued {
            if superbatch > steps.end_superbatch {
                return Err(TrainerError::DataLoadingError(DataLoadingError::TooManyBatchesReceived));
            }

            // ignore startup time from loading the first batch of data
            // because it just poisons the reported pos/sec when reading
            // from binpacked data
            if superbatch == steps.start_superbatch && curr_batch == 0 {
                superbatch_timer = Instant::now();
            }

            let lrate = lr(curr_batch, superbatch);

            if curr_batch == 0 {
                if lrate < prev_lr {
                    println!("LR dropped to {}", logger::ansi(lrate, logger::num_cs()));
                } else if lrate > prev_lr {
                    println!("LR increased to {}", logger::ansi(lrate, logger::num_cs()));
                }
            }

            prev_lr = lrate;

            let this_batch_size = batch_on_device.batch_size;
            let gf = 1.0 / this_batch_size as f32;

            batch_on_device.load_into_graph(&mut self.optimiser.graph)?;

            fn step<D: Device, G: GraphLike<D>, S: OptimiserState<D>>(
                optim: &mut Optimiser<D, G, S>,
                gradient_factor: f32,
                learning_rate: f32,
            ) -> Result<(), OperationError<D::DeviceError>> {
                optim.graph.execute_fn("zero_grads")?;
                optim.graph.execute_fn("forward")?;
                optim.graph.execute_fn("backward")?;
                optim.update(gradient_factor, learning_rate)
            }

            step(&mut self.optimiser, gf, lrate).map_err(TrainerError::GradientCalculationError)?;

            if let Ok(next_batch) = receiver.recv() {
                batch_on_device.load_new_data(&next_batch).map_err(TrainerError::Unexpected)?;
            } else {
                batch_queued = false;
            }

            let error = self.optimiser.graph.get_output_value().unwrap() / this_batch_size as f32;

            running_loss += error;
            superbatch_positions += this_batch_size;

            if curr_batch % schedule.log_rate == 0 {
                logger::report_superbatch_progress(
                    superbatch,
                    steps.batches_per_superbatch,
                    curr_batch,
                    &superbatch_timer,
                    superbatch_positions,
                );
            }

            curr_batch += 1;

            batch_callback(self, superbatch, curr_batch, error);

            if curr_batch % steps.batches_per_superbatch == 0 {
                let error = running_loss / steps.batches_per_superbatch as f32;
                running_loss = 0.0;

                let total_time = timer.elapsed().as_secs_f32();
                let sb_time = superbatch_timer.elapsed().as_secs_f32();

                logger::report_superbatch_finished(superbatch, error, sb_time, total_time, superbatch_positions);
                logger::report_time_left(steps, superbatch, total_time);

                superbatch_callback(self, superbatch);

                superbatch += 1;
                curr_batch = 0;
                superbatch_positions = 0;
                superbatch_timer = Instant::now();
            }
        }

        let total_time = timer.elapsed().as_secs();
        let (hours, minutes, seconds) = logger::seconds_to_hms(total_time as u32);

        println!(
            "Total Training Time: {}h {}m {}s",
            logger::ansi(hours, logger::num_cs()),
            logger::ansi(minutes, logger::num_cs()),
            logger::ansi(seconds, logger::num_cs()),
        );

        dataloader.join().unwrap()?;

        Ok(())
    }
}
