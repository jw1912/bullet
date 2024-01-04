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

impl<'a> LocalSettings<'a> {
    pub fn display(&self, esc: &str) {
        println!("Threads        : {}", ansi!(self.threads, 31, esc));
        println!("Data File Path : {}", ansi!(self.data_file_path, "32;1", esc));
        println!("Positions      : {}", ansi!(self.output_directory, 31, esc));
    }
}

impl<T: inputs::InputType> Trainer<T> {
    pub fn run(&mut self, schedule: &TrainingSchedule, settings: &LocalSettings) {
        training::run::<T>(
            self,
            schedule,
            settings,
        );
    }
}
