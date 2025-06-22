pub mod logger;
mod preparer;
pub mod save;
pub mod schedule;
pub mod settings;

use bullet_core::optimiser::{Optimiser, OptimiserState};
pub use preparer::DataPreparer;
use save::SavedFormat;
use schedule::{lr::LrScheduler, wdl::WdlScheduler, TrainingSchedule};
use settings::LocalSettings;

#[allow(deprecated)]
pub use crate::default;

use crate::ExecutionContext;

use std::{
    fs::File,
    io::{self, Write},
    sync::mpsc::{self, Receiver},
    time::Instant,
};

pub trait NetworkTrainer {
    type PreparedData;
    type OptimiserState: OptimiserState<ExecutionContext>;

    /// Load prepared data onto the GPU, return batch size
    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize;

    /// Trains for a single step on a batch that has been previously
    /// loaded using `load_batch`.
    fn train_on_batch(&mut self, gf: f32, lr: f32) -> f32 {
        self.optimiser().graph.synchronise().unwrap();
        self.optimiser_mut().graph.zero_grads().unwrap();

        let error = match self.optimiser_mut().graph.forward() {
            Ok(error) => error,
            Err(e) => {
                println!();
                println!("An unrecoverable error occurred:");
                println!("{e:#?}");
                std::process::exit(1);
            }
        };

        self.optimiser_mut().graph.backward().unwrap();

        self.optimiser_mut().update(gf, lr).unwrap();

        self.optimiser().graph.synchronise().unwrap();

        error
    }

    fn optimiser(&self) -> &Optimiser<ExecutionContext, Self::OptimiserState>;

    fn optimiser_mut(&mut self) -> &mut Optimiser<ExecutionContext, Self::OptimiserState>;

    fn load_from_checkpoint(&mut self, path: &str) {
        let err = self.optimiser_mut().load_from_checkpoint(&format!("{path}/optimiser_state"));
        if let Err(e) = err {
            println!();
            println!("Error loading from checkpoint:");
            println!("{e:?}");
            std::process::exit(1);
        }
    }

    fn save_to_checkpoint(&self, path: &str) {
        std::fs::create_dir(path).unwrap_or(());
        let optimiser_path = format!("{path}/optimiser_state");
        std::fs::create_dir(optimiser_path.as_str()).unwrap_or(());
        self.optimiser().write_to_checkpoint(&optimiser_path).unwrap();
    }

    fn train_custom<D1, D2, LR, WDL, F>(
        &mut self,
        preparer: &D1,
        test_preparer: &Option<D2>,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        mut callback: F,
    ) where
        D1: DataPreparer<PreparedData = Self::PreparedData> + 'static,
        D2: DataPreparer<PreparedData = Self::PreparedData> + 'static,
        LR: LrScheduler,
        WDL: WdlScheduler,
        F: FnMut(usize, &Self, &TrainingSchedule<LR, WDL>, &LocalSettings),
    {
        logger::clear_colours();

        let timer = Instant::now();
        let threads = settings.threads;
        let out_dir = settings.output_directory.to_string();
        let out_dir = out_dir.as_str();

        let mut error_record = Vec::new();
        let mut validation_record = Vec::new();

        std::fs::create_dir(out_dir).unwrap_or(());

        self.optimiser().graph.synchronise().unwrap();

        let steps = schedule.steps;

        let (sender, receiver) = mpsc::sync_channel::<D1::PreparedData>(settings.batch_queue_size);

        let dataloader =
            preparer::create_dataloader(preparer.clone(), sender, steps, schedule.wdl_scheduler.clone(), threads);

        let mut validation_freq = settings.test_set.map_or(32, |test| test.freq);

        if steps.batches_per_superbatch >= 32 && validation_freq < 32 {
            println!("Setting validation frequency to every 32 batches, come on ...");
            validation_freq = 32;
        }

        let (test_dataloader, test_receiver) = settings
            .test_set
            .map(|_| {
                let (sender, receiver) = mpsc::sync_channel::<D1::PreparedData>(2);
                let steps = schedule.steps_for_validation(validation_freq);
                let dataloader = preparer::create_dataloader(
                    test_preparer.clone().unwrap(),
                    sender,
                    steps,
                    schedule.wdl_scheduler.clone(),
                    threads,
                );
                (dataloader, receiver)
            })
            .unzip();

        let mut prev_lr = schedule.lr(0, 1);
        let mut superbatch = steps.start_superbatch;
        let mut curr_batch = 0;
        let mut superbatch_timer = Instant::now();
        let mut running_loss = 0.0;
        let mut superbatch_positions = 0;

        let mut prev32_loss = 0.0;

        while let Ok(prepared_data) = receiver.recv() {
            // ignore startup time from loading the first batch of data
            // because it just poisons the reported pos/sec
            if superbatch == steps.start_superbatch && curr_batch == 0 {
                superbatch_timer = Instant::now();
            }

            let lrate = schedule.lr(curr_batch, superbatch);

            if curr_batch == 0 {
                if lrate < prev_lr {
                    println!("LR dropped to {}", logger::ansi(lrate, logger::num_cs()));
                } else if lrate > prev_lr {
                    println!("LR increased to {}", logger::ansi(lrate, logger::num_cs()));
                }
            }

            prev_lr = lrate;

            let this_batch_size = self.load_batch(&prepared_data);
            let gf = 1.0 / this_batch_size as f32;

            let error = self.train_on_batch(gf, lrate) / this_batch_size as f32;

            running_loss += error;
            prev32_loss += error;
            superbatch_positions += this_batch_size;

            // Track test loss every freq batches.
            if curr_batch % validation_freq == 0 {
                if let Some(Ok(test_batch)) = test_receiver.as_ref().map(Receiver::recv) {
                    let this_batch_size = self.load_batch(&test_batch);
                    self.optimiser().graph.synchronise().unwrap();

                    let error = match self.optimiser_mut().graph.forward() {
                        Ok(error) => error / this_batch_size as f32,
                        Err(e) => {
                            println!();
                            println!("An unrecoverable error occurred:");
                            println!("{e:#?}");
                            std::process::exit(1);
                        }
                    };

                    validation_record.push((superbatch, curr_batch, error));
                }
            }

            if curr_batch % 128 == 0 {
                logger::report_superbatch_progress(
                    superbatch,
                    steps.batches_per_superbatch,
                    curr_batch,
                    &superbatch_timer,
                    superbatch_positions,
                );
            }

            curr_batch += 1;

            if curr_batch % 32 == 0 || (steps.batches_per_superbatch < 32 && curr_batch == steps.batches_per_superbatch)
            {
                prev32_loss /= 32.0_f32.min(steps.batches_per_superbatch as f32);

                error_record.push((superbatch, curr_batch, prev32_loss));

                prev32_loss = 0.0;
            }

            if curr_batch % steps.batches_per_superbatch == 0 {
                let error = running_loss / steps.batches_per_superbatch as f32;
                running_loss = 0.0;

                let total_time = timer.elapsed().as_secs_f32();
                let sb_time = superbatch_timer.elapsed().as_secs_f32();

                logger::report_superbatch_finished(superbatch, error, sb_time, total_time, superbatch_positions);
                logger::report_time_left(steps, superbatch, total_time);

                if schedule.should_save(superbatch) {
                    let name = format!("{}-{superbatch}", schedule.net_id());
                    let out_dir = settings.output_directory;
                    let path = format!("{out_dir}/{name}");
                    self.save_to_checkpoint(path.as_str());

                    write_losses(&format!("{path}/log.txt"), &error_record);

                    if settings.test_set.is_some() {
                        write_losses(&format!("{path}/validation-log.txt"), &validation_record);
                    }

                    println!("Saved [{}]", logger::ansi(name, 31));
                }

                callback(superbatch, self, schedule, settings);

                superbatch += 1;
                curr_batch = 0;
                prev32_loss = 0.0;
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
        if let Some(h) = test_dataloader {
            if !h.is_finished() {
                println!("Warning: Training set exhausted but test set is not!");
            }
            h.join().unwrap();
        };
    }

    fn save_weights_portion(&self, path: &str, weights: &[SavedFormat]) -> io::Result<()> {
        let mut file = File::create(path).unwrap();
        let mut buf = Vec::new();
        let graph = &self.optimiser().graph;

        for fmt in weights {
            buf.extend_from_slice(&fmt.write_to_byte_buffer(graph)?);
        }

        file.write_all(&buf)?;

        Ok(())
    }
}

fn write_losses(path: &str, error_record: &[(usize, usize, f32)]) {
    use std::io::Write;

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).expect("Opening log file failed!"));
    for (superbatch, batch, loss) in error_record {
        writeln!(writer, "{superbatch},{batch},{loss}",).expect("Writing to log file failed!");
    }
}
