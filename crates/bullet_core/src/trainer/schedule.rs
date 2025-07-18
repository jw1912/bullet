use super::logger::ansi;

#[derive(Clone, Copy, Debug)]
pub struct TrainingSteps {
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub start_superbatch: usize,
    pub end_superbatch: usize,
}

impl TrainingSteps {
    pub fn display(&self) {
        println!("Batch Size             : {}", ansi(self.batch_size, 31));
        println!("Batches / Superbatch   : {}", ansi(self.batches_per_superbatch, 31));
        println!("Positions / Superbatch : {}", ansi(self.batches_per_superbatch * self.batch_size, 31));
        println!("Start Superbatch       : {}", ansi(self.start_superbatch, 31));
        println!("End Superbatch         : {}", ansi(self.end_superbatch, 31));
    }
}

pub struct TrainingSchedule<'a> {
    pub steps: TrainingSteps,
    pub lr_schedule: Box<dyn Fn(usize, usize) -> f32 + 'a>,
    pub log_rate: usize,
}
