mod default;
pub mod logger;
mod preparer;
pub mod schedule;
pub mod settings;

pub use default::Trainer;
pub use preparer::DataPreparer;

use std::{sync::mpsc, time::Instant};

use crate::{backend::util, lr::LrScheduler, optimiser::Optimiser, wdl::WdlScheduler, LocalSettings, TrainingSchedule};

pub trait NetworkTrainer {
    type PreparedData;
    type Optimiser: Optimiser;

    /// Load prepared data onto the GPU, return batch size
    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize;

    /// Trains for a single step on a batch that has been previously
    /// loaded using `load_batch`.
    fn train_on_batch(&mut self, gf: f32, lr: f32) -> f32 {
        util::device_synchronise();
        self.optimiser_mut().graph_mut().zero_grads();

        let error = self.optimiser_mut().graph_mut().forward();

        self.optimiser_mut().graph_mut().backward();

        self.optimiser_mut().update(gf, lr);

        util::device_synchronise();
        util::panic_if_device_error("Something went wrong!");

        error
    }

    fn optimiser(&self) -> &Self::Optimiser;

    fn optimiser_mut(&mut self) -> &mut Self::Optimiser;

    fn train_custom<D, LR, WDL, F>(
        &mut self,
        preparer: &D,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        mut callback: F,
    ) where
        D: DataPreparer<PreparedData = Self::PreparedData> + 'static,
        LR: LrScheduler,
        WDL: WdlScheduler,
        F: FnMut(usize, &Self, &TrainingSchedule<LR, WDL>, &LocalSettings),
    {
        logger::clear_colours();

        let timer = Instant::now();
        let threads = settings.threads;
        let out_dir = settings.output_directory.to_string();
        let out_dir = out_dir.as_str();

        std::fs::create_dir(out_dir).unwrap_or(());

        util::device_synchronise();

        let steps = schedule.steps;
        let pos_per_sb = steps.batch_size * steps.batches_per_superbatch;

        let (sender, reciever) = mpsc::sync_channel::<D::PreparedData>(settings.batch_queue_size);

        let dataloader =
            preparer::create_dataloader(preparer.clone(), sender, steps, schedule.wdl_scheduler.clone(), threads);

        let mut prev_lr = schedule.lr(0, 1);
        let mut superbatch = steps.start_superbatch;
        let mut curr_batch = 0;
        let mut superbatch_timer = Instant::now();
        let mut running_loss = 0.0;

        while let Ok(prepared_data) = reciever.recv() {
            let lrate = schedule.lr(curr_batch, superbatch);

            if lrate != prev_lr {
                println!("LR Dropped to {}", logger::ansi(lrate, logger::num_cs()));
            }

            prev_lr = lrate;

            let this_batch_size = self.load_batch(&prepared_data);
            let gf = 1.0 / this_batch_size as f32;

            let error = self.train_on_batch(gf, lrate);

            running_loss += error / this_batch_size as f32;

            if curr_batch % 128 == 0 {
                logger::report_superbatch_progress(
                    superbatch,
                    steps.batch_size,
                    steps.batches_per_superbatch,
                    curr_batch,
                    &superbatch_timer,
                );
            }

            curr_batch += 1;

            if curr_batch % steps.batches_per_superbatch == 0 {
                let error = running_loss / steps.batches_per_superbatch as f32;
                running_loss = 0.0;

                let total_time = timer.elapsed().as_secs_f32();
                let sb_time = superbatch_timer.elapsed().as_secs_f32();

                logger::report_superbatch_finished(superbatch, error, sb_time, total_time, pos_per_sb);
                logger::report_time_left(steps, superbatch, total_time);

                if schedule.should_save(superbatch) {
                    let name = format!("{}-{superbatch}", schedule.net_id());
                    let out_dir = settings.output_directory;
                    let path = format!("{out_dir}/{name}");
                    std::fs::create_dir(path.as_str()).unwrap_or(());

                    self.optimiser().write_to_checkpoint(&path);

                    println!("Saved [{}]", logger::ansi(name, 31));
                }

                callback(superbatch, self, schedule, settings);

                superbatch += 1;
                curr_batch = 0;
                superbatch_timer = Instant::now();
            }
        }

        let total_time = timer.elapsed().as_secs();
        let (hours, minutes, seconds) = logger::seconds_to_hms(total_time as u32);

        println!(
            "Estimated time remaining in training: {}h {}m {}s",
            logger::ansi(hours, logger::num_cs()),
            logger::ansi(minutes, logger::num_cs()),
            logger::ansi(seconds, logger::num_cs()),
        );

        dataloader.join().unwrap();
    }
}
