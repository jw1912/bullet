use crate::ansi;

pub struct TrainingSchedule {
    pub net_id: String,
    pub start_epoch: usize,
    pub num_epochs: usize,
    pub wdl_scheduler: WdlScheduler,
    pub lr_scheduler: LrScheduler,
    pub save_rate: usize,
}

impl TrainingSchedule {
    pub fn net_id(&self) -> String {
        self.net_id.clone()
    }

    pub fn num_epochs(&self) -> usize {
        self.num_epochs
    }

    pub fn should_save(&self, epoch: usize) -> bool {
        epoch % self.save_rate == 0
    }

    pub fn lr(&self) -> f32 {
        self.lr_scheduler.lr()
    }

    pub fn wdl(&self, epoch: usize) -> f32 {
        self.wdl_scheduler.blend(epoch, self.num_epochs)
    }

    pub fn update(&mut self, epoch: usize, num_cs: i32, esc: &str) {
        self.lr_scheduler.adjust(epoch, num_cs, esc);
    }
}

pub struct LrScheduler {
    val: f32,
    gamma: f32,
    scheduler: LrSchedulerType,
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

pub enum LrSchedulerType {
    /// Drop once, by a factor of `gamma`.
    Drop(usize),
    /// Drop every N epochs by a factor of `gamma`.
    /// Exponential is here with `step = 1`.
    Step(usize),
}

impl LrScheduler {
    pub fn new(val: f32, gamma: f32, scheduler: LrSchedulerType) -> Self {
        Self {
            val,
            gamma,
            scheduler,
        }
    }

    pub fn lr(&self) -> f32 {
        self.val
    }

    pub fn set_type(&mut self, scheduler: LrSchedulerType) {
        self.scheduler = scheduler;
    }

    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = gamma;
    }

    pub fn adjust(&mut self, epoch: usize, num_cs: i32, esc: &str) {
        if match self.scheduler {
            LrSchedulerType::Drop(drop) => drop == epoch,
            LrSchedulerType::Step(step) => epoch % step == 0,
        } {
            self.val *= self.gamma;
            if self.gamma != 1.0 {
                println!("LR Dropped to {}", ansi!(self.val, num_cs, esc));
            }
        }
    }

    pub fn colourful(&self, esc: &str) -> String {
        let sched = match self.scheduler {
            LrSchedulerType::Drop(x) => format!("at {}", ansi!(x, 31, esc)),
            LrSchedulerType::Step(x) => format!("every {}", ansi!(x, 31, esc)),
        };

        format!(
            "start {} gamma {} drop {} epochs",
            ansi!(self.val, 31, esc),
            ansi!(self.gamma, 31, esc),
            sched
        )
    }
}
