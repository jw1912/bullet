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
pub use trainer::{
    schedule::{LrScheduler, TrainingSchedule, WdlScheduler},
    set_cbcs, Trainer, TrainerBuilder,
};

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

pub enum TimeControl {
    Increment { time: usize, inc: usize },
    FixedNodes(usize),
}

pub struct UciOption<'a>(pub &'a str, pub &'a str);

pub struct Engine<'a> {
    pub repo: &'a str,
    pub branch: &'a str,
    pub bench: Option<usize>,
    pub net_path: Option<&'a str>,
    pub uci_options: Vec<UciOption<'a>>,
}

pub struct TestSettings<'a> {
    pub out_dir: &'a str,
    pub cutechess_path: &'a str,
    pub book_path: &'a str,
    pub num_game_pairs: usize,
    pub concurrency: usize,
    pub time_control: TimeControl,
    pub base_engine: Engine<'a>,
    pub dev_engine: Engine<'a>,
}

impl<T: inputs::InputType, U: outputs::OutputBuckets<T::RequiredDataType>> Trainer<T, U> {
    pub fn run_custom<F>(
        &mut self,
        schedule: &TrainingSchedule,
        settings: &LocalSettings,
        callback: F,
    ) where
        F: FnMut(usize, &Trainer<T, U>, &TrainingSchedule, &LocalSettings),
    {
        trainer::run::<T, U, F>(self, schedule, settings, callback);
    }

    pub fn run(&mut self, schedule: &TrainingSchedule, settings: &LocalSettings) {
        self.run_custom(
            schedule,
            settings,
            |superbatch, trainer, schedule, settings| {
                if schedule.should_save(superbatch) {
                    let name = format!("{}-{superbatch}", schedule.net_id());
                    let out_dir = settings.output_directory;
                    trainer.save(out_dir, name.clone());
                    println!("Saved [{}]", ansi(name, 31));
                }
            },
        );
    }

    pub fn run_and_test(
        &mut self,
        schedule: &TrainingSchedule,
        settings: &LocalSettings,
        testing: &TestSettings,
    ) {
        let TestSettings {
            out_dir,
            cutechess_path,
            book_path,
            num_game_pairs,
            concurrency,
            time_control,
            base_engine,
            dev_engine,
        } = testing;

        self.run_custom(
            schedule,
            settings,
            |superbatch, trainer, schedule, settings| {
                if schedule.should_save(superbatch) {
                    let name = format!("{}-{superbatch}", schedule.net_id());
                    let out_dir = settings.output_directory;
                    trainer.save(out_dir, name.clone());
                    println!("Saved [{}]", ansi(name, 31));


                }
            },
        );
    }
}
