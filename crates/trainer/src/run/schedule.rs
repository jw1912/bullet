use std::num::NonZeroUsize;

use super::logger::ansi;

#[derive(Clone, Copy, Debug)]
pub struct TrainingSteps {
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub start_superbatch: usize,
    pub end_superbatch: usize,
}

impl TrainingSteps {
    pub fn start_batch(&self) -> usize {
        self.batches_per_superbatch * (self.start_superbatch - 1)
    }

    pub fn display(&self) {
        println!("Batch Size             : {}", ansi(self.batch_size, 31));
        println!("Batches / Superbatch   : {}", ansi(self.batches_per_superbatch, 31));
        println!("Positions / Superbatch : {}", ansi(self.batches_per_superbatch * self.batch_size, 31));
        println!("Start Superbatch       : {}", ansi(self.start_superbatch, 31));
        println!("End Superbatch         : {}", ansi(self.end_superbatch, 31));
    }
}

pub struct TrainingSchedule {
    pub steps: TrainingSteps,
    pub lr_schedule: Box<dyn Fn(Step) -> f32>,
    pub log_rate: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Step {
    superbatch: NonZeroUsize,
    batch: usize,
    final_superbatch: NonZeroUsize,
    batches_per_superbatch: NonZeroUsize,
}

impl From<TrainingSteps> for Step {
    fn from(value: TrainingSteps) -> Self {
        Step::new(value.start_superbatch, 0, value.end_superbatch, value.batches_per_superbatch)
    }
}

impl Default for Step {
    fn default() -> Self {
        Step::new(1, 0, 1, 1)
    }
}

impl Step {
    pub fn new(superbatch: usize, batch: usize, final_superbatch: usize, batches_per_superbatch: usize) -> Self {
        assert!(batch < batches_per_superbatch);
        assert!(superbatch <= final_superbatch);

        Self {
            superbatch: superbatch.try_into().unwrap(),
            batch,
            final_superbatch: final_superbatch.try_into().unwrap(),
            batches_per_superbatch: batches_per_superbatch.try_into().unwrap(),
        }
    }

    pub fn finished(&self) -> bool {
        self.superbatch > self.final_superbatch
    }

    pub fn step(&mut self) {
        self.batch += 1;

        if self.batch == self.batches_per_superbatch.get() {
            self.batch = 0;
            self.superbatch = self.superbatch.checked_add(1).unwrap();
        }
    }

    pub fn total_batches(&self) -> usize {
        self.batch() + self.batches_per_superbatch() * (self.superbatch() - 1)
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn superbatch(&self) -> usize {
        self.superbatch.get()
    }

    pub fn final_superbatch(&self) -> usize {
        self.final_superbatch.get()
    }

    pub fn batches_per_superbatch(&self) -> usize {
        self.batches_per_superbatch.get()
    }
}
