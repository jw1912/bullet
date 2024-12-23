pub mod lr;
pub mod wdl;

use lr::LrScheduler;
use std::fmt::Debug;
use wdl::WdlScheduler;

use super::logger::ansi;

#[derive(Clone, Copy, Debug)]
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
}
