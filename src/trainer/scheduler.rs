use crate::ansi;

pub struct LrScheduler {
    val: f32,
    gamma: f32,
    scheduler: SchedulerType,
}

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

    pub fn adjust(&mut self, epoch: usize, num_cs: i32, esc: &str) {
        if match self.scheduler {
            SchedulerType::Drop(drop) => drop == epoch,
            SchedulerType::Step(step) => epoch % step == 0,
        } {
            self.val *= self.gamma;
            if self.gamma != 1.0 {
                println!("LR Dropped to {}", ansi!(self.val, num_cs, esc));
            }
        }
    }

    pub fn colourful(&self, esc: &str) -> String {
        let sched = match self.scheduler {
            SchedulerType::Drop(x) => format!("at {}", ansi!(x, 31, esc)),
            SchedulerType::Step(x) => format!("every {}", ansi!(x, 31, esc)),
        };

        format!(
            "start {} gamma {} drop {} epochs",
            ansi!(self.val, 31, esc),
            ansi!(self.gamma, 31, esc),
            sched
        )
    }
}
