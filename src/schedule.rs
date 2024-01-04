use crate::ansi;

pub struct TrainingSchedule {
    pub net_id: String,
    pub start_epoch: usize,
    pub end_epoch: usize,
    pub wdl_scheduler: WdlScheduler,
    pub lr_scheduler: LrScheduler,
    pub save_rate: usize,
}

impl TrainingSchedule {
    pub fn net_id(&self) -> String {
        self.net_id.clone()
    }

    pub fn should_save(&self, epoch: usize) -> bool {
        epoch % self.save_rate == 0 || epoch == self.end_epoch
    }

    pub fn lr(&self, epoch: usize) -> f32 {
        self.lr_scheduler.lr(epoch)
    }

    pub fn wdl(&self, epoch: usize) -> f32 {
        self.wdl_scheduler.blend(epoch, self.end_epoch)
    }
}

#[derive(Clone, Copy)]
pub enum LrScheduler {
    /// Constant Rate
    Constant { start: f32 },
    /// Drop once at epoch `drop`, by a factor of `gamma`.
    Drop { start: f32, gamma: f32, drop: usize },
    /// Drop every `step` epochs by a factor of `gamma`.
    Step { start: f32, gamma: f32, step: usize },
}

impl LrScheduler {
    pub fn lr(&self, epoch: usize) -> f32 {
        match *self {
            Self::Constant { start } => start,
            Self::Drop { start, gamma, drop } => {
                if epoch > drop { start * gamma } else { start }
            }
            Self::Step { start, gamma, step } => {
                let steps = epoch.saturating_sub(1) / step;
                start * gamma.powi(steps as i32)
            }
        }
    }

    pub fn colourful(&self, esc: &str) -> String {
        match *self {
            Self::Constant { start } => format!("{start}"),
            Self::Drop { start, gamma, drop } => {
                format!(
                    "start {} gamma {} drop at {} epochs",
                    ansi!(start, 31, esc),
                    ansi!(gamma, 31, esc),
                    ansi!(drop, 31, esc),
                )
            }
            Self::Step { start, gamma, step } => {
                format!(
                    "start {} gamma {} drop every {} epochs",
                    ansi!(start, 31, esc),
                    ansi!(gamma, 31, esc),
                    ansi!(step, 31, esc),
                )
            },
        }
    }
}

pub struct WdlScheduler {
    start: f32,
    end: f32,
}

impl WdlScheduler {
    pub fn new(start: f32, mut end: f32) -> Self {
        if end < 0.0 {
            end = start;
        }

        Self { start, end }
    }

    pub fn blend(&self, epoch: usize, max_epochs: usize) -> f32 {
        let grad = (self.end - self.start) / (max_epochs - 1).max(1) as f32;
        self.start + grad * (epoch - 1) as f32
    }

    pub fn start(&self) -> f32 {
        self.start
    }

    pub fn end(&self) -> f32 {
        self.end
    }
}
