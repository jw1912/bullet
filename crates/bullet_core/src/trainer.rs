pub mod dataloader;
pub mod logger;
pub mod schedule;

use std::{sync::mpsc, thread, time::Instant};

use crate::{
    backend::device::{Device, OperationError}, optimiser::{Optimiser, OptimiserState}, trainer::schedule::TrainingSchedule
};

pub enum TrainerError<D: Device> {
    DataLoadingError,
    GradientCalculationError(OperationError<D::DeviceError>),
    IoError,
}

pub struct Trainer<D: Device, O: OptimiserState<D>, S> {
    pub optimiser: Optimiser<D, O>,
    pub state: S,
}

impl<D: Device, O: OptimiserState<D>, S> Trainer<D, O, S> {
    pub fn train_custom(&mut self, schedule: TrainingSchedule) -> Result<(), TrainerError<D>> {
        logger::clear_colours();

        let timer = Instant::now();
        let out_dir = schedule.out_dir.as_str();
        let lr = schedule.lr_schedule;

        let _ = std::fs::create_dir(out_dir);

        self.optimiser.graph.synchronise().unwrap();

        let steps = schedule.steps;

        let (sender, receiver) = mpsc::sync_channel::<()>(32);

        let dataloader = thread::spawn(|| {});

        let mut prev_lr = lr(0, 1);
        let mut superbatch = steps.start_superbatch;
        let mut curr_batch = 0;
        let mut superbatch_timer = Instant::now();
        let mut running_loss = 0.0;
        let mut superbatch_positions = 0;

        let first_batch = receiver.recv().map_err(|_| TrainerError::DataLoadingError)?;

        let mut batch_queued = true;

        while batch_queued {
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

            let this_batch_size: usize = 0;
            let gf = 1.0 / this_batch_size as f32;

            fn step<D: Device, S: OptimiserState<D>>(
                optim: &mut Optimiser<D, S>,
                gradient_factor: f32,
                learning_rate: f32,
            ) -> Result<(), OperationError<D::DeviceError>> {
                optim.graph.zero_grads_non_blocking()?;
                optim.graph.forward_non_blocking()?;
                optim.graph.backward_non_blocking()?;
                optim.update(gradient_factor, learning_rate)
            }

            step(&mut self.optimiser, gf, lrate).map_err(TrainerError::GradientCalculationError)?;

            match receiver.recv() {
                Ok(_next_batch) => {todo!()},
                Err(_) => batch_queued = false,
            }

            let error = self.optimiser.graph.get_output_val().unwrap();

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

            if curr_batch % steps.batches_per_superbatch == 0 {
                let error = running_loss / steps.batches_per_superbatch as f32;
                running_loss = 0.0;

                let total_time = timer.elapsed().as_secs_f32();
                let sb_time = superbatch_timer.elapsed().as_secs_f32();

                logger::report_superbatch_finished(superbatch, error, sb_time, total_time, superbatch_positions);
                logger::report_time_left(steps, superbatch, total_time);

                if superbatch % schedule.save_rate == 0 || superbatch == steps.end_superbatch {
                    let name = format!("{}-{superbatch}", schedule.net_id);
                    let path = format!("{out_dir}/{name}");
                    self.optimiser.write_to_checkpoint(path.as_str()).map_err(|_| TrainerError::IoError)?;

                    println!("Saved [{}]", logger::ansi(name, 31));
                }

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

        dataloader.join().unwrap();

        Ok(())
    }
}
