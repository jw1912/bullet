use std::fmt::Debug;

use lr::LrScheduler;
use wdl::WdlScheduler;

use super::logger::ansi;

pub mod lr;
pub mod wdl;

pub use bullet_core::trainer::schedule::TrainingSteps;

#[derive(Clone, Debug)]
pub struct TrainingSchedule<LR: LrScheduler, WDL: WdlScheduler> {
    pub net_id: String,
    pub eval_scale: f32,
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
        println!("Net Name               : {}", ansi(self.net_id.clone(), "32;1"));
        self.steps.display();
        println!("Eval Scale             : {}", ansi(format!("{:.0}", self.eval_scale), 31));
        println!("Save Rate              : {}", ansi(self.save_rate, 31));
        println!("WDL Scheduler          : {}", self.wdl_scheduler.colourful());
        println!("LR Scheduler           : {}", self.lr_scheduler.colourful());
    }

    /// For evaluation passes, in order to ensure that we exhaust the test set at the
    /// same time as we exhaust the training set.
    pub fn steps_for_validation(&self, validation_freq: usize) -> TrainingSteps {
        let mut res = self.steps;

        res.batches_per_superbatch = self.steps.batches_per_superbatch / validation_freq;

        res
    }
}
