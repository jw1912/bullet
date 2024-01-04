mod schedule;
mod trainer;
mod training;

pub use bullet_core::inputs;
pub use bullet_tensor::Activation;
pub use schedule::{LrScheduler, TrainingSchedule, WdlScheduler};
pub use trainer::{Trainer, TrainerBuilder};
pub use training::set_cbcs;

pub struct LocalSettings<'a> {
    pub threads: usize,
    pub data_file_path: &'a str,
    pub output_directory: &'a str,
}

impl<T: inputs::InputType> Trainer<T> {
    pub fn run(&mut self, schedule: &TrainingSchedule, settings: &LocalSettings) {
        training::run::<T>(
            self,
            schedule,
            settings.threads,
            settings.data_file_path,
            settings.output_directory,
        );
    }
}
