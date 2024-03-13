use crate::ansi;

#[derive(Clone)]
pub struct TrainingSchedule {
    pub net_id: String,
    pub batch_size: usize,
    pub eval_scale: f32,
    pub batches_per_superbatch: usize,
    pub start_superbatch: usize,
    pub end_superbatch: usize,
    pub wdl_scheduler: WdlScheduler,
    pub lr_scheduler: LrScheduler,
    pub save_rate: usize,
}

impl TrainingSchedule {
    pub fn net_id(&self) -> String {
        self.net_id.clone()
    }

    pub fn should_save(&self, superbatch: usize) -> bool {
        superbatch % self.save_rate == 0 || superbatch == self.end_superbatch
    }

    pub fn lr(&self, superbatch: usize) -> f32 {
        self.lr_scheduler.lr(superbatch)
    }

    pub fn wdl(&self, superbatch: usize) -> f32 {
        self.wdl_scheduler.blend(superbatch, self.end_superbatch)
    }

    pub fn display(&self) {
        println!(
            "Scale                  : {}",
            ansi(format!("{:.0}", self.eval_scale), 31)
        );
        println!("Batch Size             : {}", ansi(self.batch_size, 31));
        println!(
            "Batches / Superbatch   : {}",
            ansi(self.batches_per_superbatch, 31)
        );
        println!(
            "Positions / Superbatch : {}",
            ansi(self.batches_per_superbatch * self.batch_size, 31)
        );
        println!(
            "Start Superbatch       : {}",
            ansi(self.start_superbatch, 31)
        );
        println!("End Superbatch         : {}", ansi(self.end_superbatch, 31));
        println!("Save Rate              : {}", ansi(self.save_rate, 31));
        println!(
            "WDL Scheduler          : {}",
            self.wdl_scheduler.colourful()
        );
        println!("LR Scheduler           : {}", self.lr_scheduler.colourful());
    }
}

#[derive(Clone, Copy)]
pub enum LrScheduler {
    /// Constant Rate
    Constant { value: f32 },
    /// Drop once at superbatch `drop`, by a factor of `gamma`.
    Drop { start: f32, gamma: f32, drop: usize },
    /// Drop every `step` superbatches by a factor of `gamma`.
    Step { start: f32, gamma: f32, step: usize },
}

impl LrScheduler {
    pub fn lr(&self, superbatch: usize) -> f32 {
        match *self {
            Self::Constant { value } => value,
            Self::Drop { start, gamma, drop } => {
                if superbatch > drop {
                    start * gamma
                } else {
                    start
                }
            }
            Self::Step { start, gamma, step } => {
                let steps = superbatch.saturating_sub(1) / step;
                start * gamma.powi(steps as i32)
            }
        }
    }

    pub fn colourful(&self) -> String {
        match *self {
            Self::Constant { value } => format!("constant {}", ansi(value, 31)),
            Self::Drop { start, gamma, drop } => {
                format!(
                    "start {} gamma {} drop at {} superbatches",
                    ansi(start, 31),
                    ansi(gamma, 31),
                    ansi(drop, 31),
                )
            }
            Self::Step { start, gamma, step } => {
                format!(
                    "start {} gamma {} drop every {} superbatches",
                    ansi(start, 31),
                    ansi(gamma, 31),
                    ansi(step, 31),
                )
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum WdlScheduler {
    Constant { value: f32 },
    Linear { start: f32, end: f32 },
}

impl WdlScheduler {
    pub fn blend(&self, superbatch: usize, max: usize) -> f32 {
        match *self {
            Self::Constant { value } => value,
            Self::Linear { start, end } => {
                let grad = (end - start) / (max - 1).max(1) as f32;
                start + grad * (superbatch - 1) as f32
            }
        }
    }

    pub fn colourful(&self) -> String {
        match *self {
            Self::Constant { value } => format!("constant {}", ansi(value, 31)),
            Self::Linear { start, end } => {
                format!(
                    "linear taper start {} end {}",
                    ansi(start, 31),
                    ansi(end, 31)
                )
            }
        }
    }
}
