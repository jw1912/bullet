use std::fmt::Debug;

use lr::LrScheduler;
use wdl::WdlScheduler;

use super::logger::ansi;

pub mod lr;
pub mod wdl;

#[derive(Clone, Debug)]
pub struct TrainingSteps {
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub start_superbatch: usize,
    pub end_superbatch: usize,
}

impl TrainingSteps {
    fn display(&self) {
        println!("Batch Size             : {}", ansi(self.batch_size, 31));
        println!("Batches / Superbatch   : {}", ansi(self.batches_per_superbatch, 31));
        println!("Positions / Superbatch : {}", ansi(self.batches_per_superbatch * self.batch_size, 31));
        println!("Start Superbatch       : {}", ansi(self.start_superbatch, 31));
        println!("End Superbatch         : {}", ansi(self.end_superbatch, 31));
    }
}

#[derive(Clone, Debug)]
pub struct TrainingSchedule<LR: LrScheduler, WDL: WdlScheduler> {
    /// Name of the training run.
    pub net_id: String,
    /// Scalar to divide evaluations from data by.
    pub eval_scale: f32,
    /// Regularisation on L1, experiment from Beserk.
    pub ft_regularisation: f32,
    pub steps: TrainingSteps,
    pub wdl_scheduler: WDL,
    pub lr_scheduler: LR,
    pub save_rate: usize,
}

impl<LR: LrScheduler, WDL: WdlScheduler> TrainingSchedule<LR, WDL> {
    pub fn net_id(&self) -> String {
        self.net_id.clone()
    }

    pub fn should_save(&self, superbatch: usize) -> bool {
        superbatch % self.save_rate == 0 || superbatch == self.steps.end_superbatch
    }

    pub fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        self.lr_scheduler.lr(batch, superbatch)
    }

    pub fn wdl(&self, batch: usize, superbatch: usize) -> f32 {
        self.wdl_scheduler.blend(batch, superbatch, self.steps.end_superbatch)
    }

    pub fn display(&self) {
        println!("Scale                  : {}", ansi(format!("{:.0}", self.eval_scale), 31));
        println!("1 / FT Regularisation  : {}", ansi(format!("{:.0}", 1.0 / self.ft_regularisation), 31));
        self.steps.display();
        println!("Save Rate              : {}", ansi(self.save_rate, 31));
        println!("WDL Scheduler          : {}", self.wdl_scheduler.colourful());
        println!("LR Scheduler           : {}", self.lr_scheduler.colourful());
    }

    /// For evaluation passes, in order to ensure that we exhaust the test set at the
    /// same time as we exhaust the training set.
    pub fn for_validation(&self, validation_freq: usize) -> Self {
        let mut res = self.clone();

        res.steps.batches_per_superbatch = self.steps.batches_per_superbatch / validation_freq;

        res
    }
}
