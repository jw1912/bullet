mod backend;
pub mod inputs;
mod loader;
pub mod outputs;
mod rng;
pub mod tensor;
mod trainer;
pub mod util;

use trainer::ansi;

pub use bulletformat as format;
pub use rng::Rand;
pub use trainer::{Trainer, TrainerBuilder, set_cbcs, schedule::{LrScheduler, TrainingSchedule, WdlScheduler}};

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    CReLU,
    SCReLU,
}

pub struct LocalSettings<'a> {
    pub threads: usize,
    pub data_file_paths: Vec<&'a str>,
    pub output_directory: &'a str,
}

impl<'a> LocalSettings<'a> {
    pub fn display(&self) {
        println!("Threads                : {}", ansi(self.threads, 31));
        for file_path in self.data_file_paths.iter() {
            println!("Data File Path         : {}", ansi(file_path, "32;1"));
        }
        println!(
            "Output Path            : {}",
            ansi(self.output_directory, "32;1")
        );
    }
}

impl<T: inputs::InputType, U: outputs::OutputBuckets<T::RequiredDataType>> Trainer<T, U> {
    pub fn run(&mut self, schedule: &TrainingSchedule, settings: &LocalSettings) {
        trainer::run::<T, U>(self, schedule, settings);
    }
}
