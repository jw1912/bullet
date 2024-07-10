mod backend;
pub mod inputs;
mod loader;
pub mod optimiser;
pub mod outputs;
pub mod tensor;
mod trainer;
pub mod util;

use std::{
    fs::{self, File},
    io::Write,
    process::{Command, Stdio},
};

use inputs::InputType;
use optimiser::Optimiser;
use outputs::OutputBuckets;
use trainer::ansi;

pub use bulletformat as format;
pub use trainer::{
    schedule::{lr, wdl, Loss, TrainingSchedule},
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
    pub test_file_path: Option<&'a str>,
    pub output_directory: &'a str,
}

impl<'a> LocalSettings<'a> {
    pub fn display(&self) {
        println!("Threads                : {}", ansi(self.threads, 31));
        for file_path in self.data_file_paths.iter() {
            println!("Data File Path         : {}", ansi(file_path, "32;1"));
        }
        println!("Output Path            : {}", ansi(self.output_directory, "32;1"));
    }
}

#[derive(Clone, Copy)]
pub enum TimeControl {
    Increment { time: f32, inc: f32 },
    FixedNodes(usize),
}

#[derive(Clone, Copy)]
pub enum OpeningBook<'a> {
    Epd(&'a str),
    Pgn(&'a str),
}

#[derive(Clone)]
pub struct UciOption<'a>(pub &'a str, pub &'a str);

#[derive(Clone)]
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
    pub book_path: OpeningBook<'a>,
    pub num_game_pairs: usize,
    pub concurrency: usize,
    pub time_control: TimeControl,
    pub base_engine: Engine<'a>,
    pub dev_engine: Engine<'a>,
}

impl<T: InputType, U: OutputBuckets<T::RequiredDataType>, O: Optimiser> Trainer<T, U, O> {
    pub fn run_custom<F, LR, WDL>(
        &mut self,
        schedule: &TrainingSchedule<O::AdditionalOptimiserParams, LR, WDL>,
        settings: &LocalSettings,
        callback: F,
    ) where
        F: FnMut(usize, &Trainer<T, U, O>, &TrainingSchedule<O::AdditionalOptimiserParams, LR, WDL>, &LocalSettings),
        LR: lr::LrScheduler,
        WDL: wdl::WdlScheduler,
    {
        trainer::run::<T, U, O, F, LR, WDL>(self, schedule, settings, callback);
    }

    pub fn run<LR, WDL>(
        &mut self,
        schedule: &TrainingSchedule<O::AdditionalOptimiserParams, LR, WDL>,
        settings: &LocalSettings,
    ) where
        LR: lr::LrScheduler,
        WDL: wdl::WdlScheduler,
    {
        self.run_custom(schedule, settings, |superbatch, trainer, schedule, settings| {
            if schedule.should_save(superbatch) {
                let name = format!("{}-{superbatch}", schedule.net_id());
                let out_dir = settings.output_directory;
                trainer.save(out_dir, name.clone());
                println!("Saved [{}]", ansi(name, 31));
            }
        });
    }

    pub fn run_and_test<LR, WDL>(
        &mut self,
        schedule: &TrainingSchedule<O::AdditionalOptimiserParams, LR, WDL>,
        settings: &LocalSettings,
        testing: &TestSettings<'static>,
    ) where
        LR: lr::LrScheduler,
        WDL: wdl::WdlScheduler,
    {
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

        let output = Command::new(cutechess_path).arg("--version").output().expect("Could not start cutechess!");

        assert!(output.status.success(), "Could not start cutechess!");

        let bpath = match book_path {
            OpeningBook::Epd(path) => path,
            OpeningBook::Pgn(path) => path,
        };

        File::open(bpath).expect("Could not find opening book!");

        fs::create_dir(out_dir).expect("The output directory already exists!");

        fs::create_dir(format!("{out_dir}/nets")).expect("Something went very wrong!");

        let stats_path = format!("{out_dir}/stats.txt");
        let sched_path = format!("{out_dir}/schedule.txt");

        File::create(stats_path.as_str()).expect("Couldn't create stats file!");
        File::create(sched_path.as_str()).expect("Couldn't create schedule file!");

        let mut sched =
            fs::OpenOptions::new().write(true).open(sched_path.as_str()).expect("Couldn't open sschedule file!");
        writeln!(&mut sched, "{schedule:#?}").expect("Couldn't write schedule to file!");

        let base_path_string = format!("{out_dir}/base_engine");
        let dev_path_string = format!("{out_dir}/dev_engine");

        let base_path = base_path_string.as_str();
        let dev_path = dev_path_string.as_str();

        let base_exe_path = format!("{base_path_string}/base_engine.exe");

        clone(base_engine, base_path);

        println!("# [Building {}/{}]", base_engine.repo, base_engine.branch);
        build(base_engine, base_path, "../base_engine/base_engine", None);

        println!("# [Running Bench]");
        bench(base_engine, base_exe_path.as_str(), true);
        println!("# [Bench Successful]");

        clone(dev_engine, dev_path);

        let mut handles = Vec::new();

        self.run_custom(schedule, settings, |superbatch, trainer, schedule, settings| {
            if schedule.should_save(superbatch) {
                let name = format!("{}-{superbatch}", schedule.net_id());
                trainer.save(settings.output_directory, name.clone());
                println!("Saved [{}]", ansi(name.as_str(), 31));
            }

            // run test
            if superbatch % test_rate == 0 || superbatch == schedule.end_superbatch {
                let name = format!("{}-{superbatch}", schedule.net_id());
                trainer.save(format!("{out_dir}/nets").as_str(), name.clone());

                println!("Testing [{}]", ansi(name.as_str(), 31));

                let base = base_engine.clone();
                let dev = dev_engine.clone();
                let dpath = dev_path_string.clone();
                let rel_dev_path = format!("../nets/{name}/{name}");
                let rel_net_path = format!("../nets/{name}/{name}.bin");
                let dev_exe_path = format!("{out_dir}/nets/{name}/{name}");
                let base_exe_path = base_exe_path.clone();
                let cc_path = cutechess_path.to_string();
                let num_game_pairs = *num_game_pairs;
                let concurrency = *concurrency;
                let time_control = *time_control;
                let book_path = *book_path;
                let stats_path = stats_path.clone();

                let handle = std::thread::spawn(move || {
                    build(&dev, dpath.as_str(), rel_dev_path.as_str(), Some(rel_net_path.as_str()));

                    bench(&dev, dev_exe_path.as_str(), false);

                    let mut cc = Command::new(cc_path);

                    cc.arg("-engine").arg(format!("cmd={dev_exe_path}"));

                    for UciOption(name, value) in dev.uci_options {
                        cc.arg(format!("option.{name}={value}"));
                    }

                    cc.arg("-engine").arg(format!("cmd={base_exe_path}"));

                    for UciOption(name, value) in base.uci_options {
                        cc.arg(format!("option.{name}={value}"));
                    }

                    cc.args(["-each", "proto=uci", "timemargin=20"]);

                    match time_control {
                        TimeControl::FixedNodes(nodes) => {
                            cc.arg("tc=inf").arg(format!("nodes={nodes}"));
                        }
                        TimeControl::Increment { time, inc } => {
                            cc.arg(format!("tc={time}+{inc}"));
                        }
                    }

                    cc.args(["-games", "2"]);

                    cc.arg("-rounds").arg(num_game_pairs.to_string());

                    cc.args(["-repeat", "2"]);

                    cc.arg("-concurrency").arg(concurrency.to_string());

                    cc.args(["-openings", "policy=round", "order=random"]);

                    match book_path {
                        OpeningBook::Epd(path) => {
                            cc.arg(format!("file={path}")).arg("format=epd");
                        }
                        OpeningBook::Pgn(path) => {
                            cc.arg(format!("file={path}")).arg("format=pgn");
                        }
                    }

                    cc.args(["-resign", "movecount=3", "score=400", "twosided=true"]);
                    cc.args(["-draw", "movenumber=40", "movecount=8", "score=10"]);

                    cc.stdout(Stdio::piped());

                    let output = cc.spawn().expect("Couldn't launch cutechess games!");

                    let output = output.wait_with_output().expect("Couldn't wait on output!");

                    let stdout = String::from_utf8(output.stdout).expect("Couldn't parse stdout!");

                    let mut split = stdout.split("Elo difference: ");

                    let line = split.nth(1).unwrap();

                    let mut split_line = line.split(',');
                    let elo_segment = split_line.next().unwrap().split_whitespace().collect::<Vec<_>>();

                    if let [elo, "+/-", err] = elo_segment[..] {
                        let mut file = fs::OpenOptions::new()
                            .append(true)
                            .open(stats_path.as_str())
                            .expect("Couldn't open stats path!");

                        writeln!(file, "{superbatch}, {elo}, {err}").expect("Couldn't write to file!");
                    } else {
                        panic!("Couldn't find elo line!");
                    }
                });

                handles.push(handle);
            }
        });

        println!("# [Waiting for Tests]");
        for handle in handles {
            if let Err(err) = handle.join() {
                println!("{err:?}");
            }
        }
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
    let mut build_base = Command::new("make");

    build_base.current_dir(inp_path).arg(format!("EXE={out_path}"));

    if let Some(net_path) = override_net {
        build_base.arg(format!("EVALFILE={}", net_path));
    } else if let Some(net_path) = engine.net_path {
        build_base.arg(format!("EVALFILE={}", net_path));
    }

    let output = build_base.output().expect("Failed to build engine!");

    assert!(output.status.success(), "Failed to build engine!");
}

fn bench(engine: &Engine, path: &str, check_match: bool) {
    let mut bench = Command::new(path);

    let output = bench.arg("bench").output().expect("Failed to run bench on engine!");

    assert!(output.status.success(), "Failed to run bench on engine!");

    if check_match {
        if let Some(bench) = engine.bench {
            let out = String::from_utf8(output.stdout).expect("Could not parse bench output!");

            let split = out.split_whitespace();

            let mut found = false;

            let mut prev = "what";
            for word in split {
                if word == "nodes" {
                    found = true;
                    assert_eq!(bench, prev.parse().expect("Could not parse bench output!"), "Bench did not match!");

                    break;
                }

                prev = word;
            }

            assert!(found, "Could not find bench!");
        }
    }
}
