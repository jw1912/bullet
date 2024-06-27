use std::fmt::Debug;

use lr::LrScheduler;
use wdl::WdlScheduler;

use crate::ansi;

pub mod lr;
pub mod wdl;

#[derive(Clone, Debug)]
pub struct TrainingSchedule<O: Clone + std::fmt::Debug + Sync + Send, LR: LrScheduler, WDL: WdlScheduler> {
    pub net_id: String,
    pub eval_scale: f32,
    pub ft_regularisation: f32,
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub start_superbatch: usize,
    pub end_superbatch: usize,
    pub wdl_scheduler: WDL,
    pub lr_scheduler: LR,
    pub loss_function: Loss,
    pub save_rate: usize,
    pub optimiser_settings: O,
}

impl<O: Clone + std::fmt::Debug + Sync + Send, LR: LrScheduler, WDL: WdlScheduler> TrainingSchedule<O, LR, WDL> {
    pub fn net_id(&self) -> String {
        self.net_id.clone()
    }

    pub fn should_save(&self, superbatch: usize) -> bool {
        superbatch % self.save_rate == 0 || superbatch == self.end_superbatch
    }

    pub fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        self.lr_scheduler.lr(batch, superbatch)
    }

    pub fn wdl(&self, batch: usize, superbatch: usize) -> f32 {
        self.wdl_scheduler.blend(batch, superbatch, self.end_superbatch)
    }

    pub fn display(&self) {
        println!("Scale                  : {}", ansi(format!("{:.0}", self.eval_scale), 31));
        println!("1 / FT Regularisation  : {}", ansi(format!("{:.0}", 1.0 / self.ft_regularisation), 31));
        println!("Batch Size             : {}", ansi(self.batch_size, 31));
        println!("Batches / Superbatch   : {}", ansi(self.batches_per_superbatch, 31));
        println!("Positions / Superbatch : {}", ansi(self.batches_per_superbatch * self.batch_size, 31));
        println!("Start Superbatch       : {}", ansi(self.start_superbatch, 31));
        println!("End Superbatch         : {}", ansi(self.end_superbatch, 31));
        println!("Save Rate              : {}", ansi(self.save_rate, 31));
        println!("WDL Scheduler          : {}", self.wdl_scheduler.colourful());
        println!("LR Scheduler           : {}", self.lr_scheduler.colourful());
    }

    pub fn power(&self) -> f32 {
        match self.loss_function {
            Loss::SigmoidMSE => 2.0,
            Loss::SigmoidMPE(x) => x,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Loss {
    SigmoidMSE,
    SigmoidMPE(f32),
}
