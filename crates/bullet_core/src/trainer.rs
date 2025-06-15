pub mod dataloader;
pub mod logger;
pub mod schedule;

use std::{sync::mpsc, thread, time::Instant};

use crate::{
    backend::device::Device,
    optimiser::{Optimiser, OptimiserState},
    trainer::schedule::TrainingSchedule,
};

pub enum TrainerError {
    DataLoadingError,
    GradientCalculationError,
    IoError,
}

pub struct Trainer<D: Device, O: OptimiserState<D>, S> {
    pub optimiser: Optimiser<D, O>,
    pub state: S,
}

impl<D: Device, O: OptimiserState<D>, S> Trainer<D, O, S> {
    pub fn train_custom(&mut self, schedule: TrainingSchedule) -> Result<(), TrainerError> {
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

        while let Ok(prepared_data) = receiver.recv() {
            // ignore startup time from loading the first batch of data
            // because it just poisons the reported pos/sec
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

            //let this_batch_size = self.load_batch(&prepared_data);
            //let gf = 1.0 / this_batch_size as f32;

            let error = 0.0; //self.train_on_batch(gf, lrate) / this_batch_size as f32;

            running_loss += error;
            //superbatch_positions += this_batch_size;

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
