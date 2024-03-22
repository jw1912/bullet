mod backend;
pub mod inputs;
mod loader;
pub mod outputs;
mod rng;
pub mod tensor;
mod trainer;
pub mod util;

use std::{process::Command, fs::{self, File}};

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
    pub test_rate: usize,
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
            test_rate,
            out_dir,
            cutechess_path,
            book_path,
            num_game_pairs,
            concurrency,
            time_control,
            base_engine,
            dev_engine,
        } = testing;

        assert_eq!(schedule.save_rate % test_rate, 0, "Save Rate should divide Test Rate!");

        let output = Command::new(cutechess_path)
            .arg("--help")
            .output()
            .expect("Could not start cutechess!");

        assert!(output.status.success(), "Could not start cutechess!");

        File::open(book_path)
            .expect("Could not find opening book!");

        fs::create_dir(out_dir)
            .expect("The output directory already exists!");

        let base_path = format!("{out_dir}/base_engine");
        let dev_path = format!("{out_dir}/dev_engine");

        let base_path = base_path.as_str();
        let dev_path = dev_path.as_str();

        let base_exe_path = format!("{out_dir}/base_engine_exe.exe");

        clone(base_engine, base_path);

        build(base_engine, base_path, "../base_engine_exe", None);

        bench(base_engine, base_exe_path, true);

        clone(dev_engine, dev_path);

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

fn clone(engine: &Engine, out_dir: &str) {
    println!("# [Cloning {}/{}]", engine.repo, engine.branch);

    let status = Command::new("git")
        .arg("clone")
        .arg(engine.repo)
        .arg(out_dir)
        .arg("--branch")
        .arg(engine.branch)
        .arg("--depth=1")
        .status()
        .expect("Failed to clone engine!");

    assert!(status.success(), "Failed to clone engine!")
}

fn build(engine: &Engine, inp_path: &str, out_path: &str, override_net: Option<&str>) {
    println!("# [Building {}/{}]", engine.repo, engine.branch);

    let mut build_base = Command::new("make");

    build_base
        .current_dir(inp_path)
        .arg(format!("EXE={out_path}"));

    if let Some(net_path) = override_net {
        build_base.arg(format!("EVALFILE={}", net_path));
    } else if let Some(net_path) = engine.net_path {
            build_base.arg(format!("EVALFILE={}", net_path));
    }

    let output = build_base
        .output()
        .expect("Failed to build engine!");

    assert!(output.status.success(), "Failed to build engine!");
}

fn bench(engine: &Engine, path: String, check_match: bool) {
    println!("# [Running Bench]");

    let mut bench = Command::new(path);

    let output = bench.arg("bench")
        .output()
        .expect("Failed to run bench on engine!");

    assert!(output.status.success(), "Failed to run bench on engine!");

    if check_match {
        if let Some(bench) = engine.bench {
            let out = String::from_utf8(output.stdout)
                .expect("Could not parse bench output!");

            let split = out.split_whitespace();

            let mut found = false;

            let mut prev = "what";
            for word in split {
                if word == "nodes" {
                    found = true;
                    assert_eq!(
                        bench,
                        prev.parse().expect("Could not parse bench output!"),
                        "Bench did not match!"
                    );

                    break;
                }

                prev = word;
            }

            assert!(found, "Could not find bench!");
        }
    }

    println!("# [Bench Successful]");
}
