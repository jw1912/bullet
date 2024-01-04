mod schedule;
mod trainer;
mod training;

pub use bullet_core::inputs;
pub use bullet_tensor::Activation;
pub use schedule::{LrScheduler, TrainingSchedule, WdlScheduler};
pub use trainer::{Trainer, TrainerBuilder};
pub use training::set_cbcs;

impl<T: inputs::InputType> Trainer<T> {
    pub fn run(&mut self, schedule: &TrainingSchedule, threads: usize, file: &str, out_dir: &str) {
        training::run::<T>(self, schedule, threads, file, out_dir);
    }
}
