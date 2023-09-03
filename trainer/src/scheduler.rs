use crate::ansi;

pub struct LrScheduler {
    val: f32,
    gamma: f32,
    scheduler: SchedulerType,
}

impl std::fmt::Display for LrScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "start {} gamma {} schedule {:?}",
            ansi!(self.val, 31), ansi!(self.gamma, 31), self.scheduler
        )
    }
}

#[derive(Debug)]
pub enum SchedulerType {
    /// Drop once, by a factor of `gamma`.
    Drop(usize),
    /// Drop every N epochs by a factor of `gamma`.
    /// Exponential is here with `step = 1`.
    Step(usize),
}

impl LrScheduler {
    pub fn new(val: f32, gamma: f32, scheduler: SchedulerType) -> Self {
        Self {
            val,
            gamma,
            scheduler,
        }
    }

    pub fn lr(&self) -> f32 {
        self.val
    }

    pub fn set_type(&mut self, scheduler: SchedulerType) {
        self.scheduler = scheduler;
    }

    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = gamma;
    }

    pub fn adjust(&mut self, epoch: usize) {
        if match self.scheduler {
            SchedulerType::Drop(drop) => drop == epoch,
            SchedulerType::Step(step) => epoch % step == 0,
        } {
            self.val *= self.gamma;
            if self.gamma != 1.0 {
                println!("LR Dropped to {}", ansi!(self.val));
            }
        }
    }
}
