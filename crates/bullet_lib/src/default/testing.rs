use std::{
    fmt::Display,
    fs::{self, File},
    io::{self, Write},
    process::Command,
    thread::{self, JoinHandle},
};

use crate::trainer::schedule::{lr::LrScheduler, wdl::WdlScheduler, TrainingSchedule};

use super::{
    gamerunner::{self, GameRunnerArgs, GameRunnerPathInternal},
    logger,
};

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

impl Display for UciOption<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "option.{}={}", self.0, self.1)
    }
}

#[derive(Clone, Copy)]
pub enum GameRunnerPath<'a> {
    CuteChess(&'a str),
    FastChess(&'a str),
}

impl GameRunnerPath<'_> {
    fn as_internal(&self) -> GameRunnerPathInternal {
        match self {
            GameRunnerPath::CuteChess(x) => GameRunnerPathInternal::CuteChess(x.to_string()),
            GameRunnerPath::FastChess(x) => GameRunnerPathInternal::FastChess(x.to_string()),
        }
    }
}

#[derive(Clone)]
pub struct Engine<'a, T: EngineType> {
    /// URL of git repository to clone from.
    pub repo: &'a str,
    /// Branch of git repository to clone.
    pub branch: &'a str,
    /// Optional expected bench to verify against.
    pub bench: Option<usize>,
    /// Path to network file to be used.
    pub net_path: Option<&'a str>,
    /// Any UCI options that should be passed.
    pub uci_options: Vec<UciOption<'a>>,
    /// Dictates how the engine is built and benched.
    pub engine_type: T,
}

pub trait EngineType: Sized {
    fn build(&self, repo_path: &str, exe_output_path: &str, override_net: Option<&str>) -> Result<(), String>;

    fn bench(&self, engine_exe_path: &str) -> Result<usize, String>;
}

pub struct TestSettings<'a, T: EngineType> {
    /// Test every `test_rate` superbatches.
    pub test_rate: usize,
    /// Directory to use for testing (MUST NOT EXIST CURRENTLY).
    pub out_dir: &'a str,
    /// Path to gamerunner executable.
    pub gamerunner_path: GameRunnerPath<'a>,
    /// Path to opening book.
    pub book_path: OpeningBook<'a>,
    /// Number of game pairs to play.
    pub num_game_pairs: usize,
    /// Number of games to run in parallel.
    pub concurrency: usize,
    /// Time control to run games at.
    pub time_control: TimeControl,
    /// Base engine, must provide own net.
    pub base_engine: Engine<'a, T>,
    /// Dev engine, will be given newly trained nets.
    pub dev_engine: Engine<'a, T>,
}

impl<T: EngineType> TestSettings<'_, T> {
    pub fn setup<LR: LrScheduler, WDL: WdlScheduler>(&self, schedule: &TrainingSchedule<LR, WDL>) {
        let output = gamerunner::GameRunnerCommand::health_check(&self.gamerunner_path.as_internal());

        assert!(output.status.success(), "Could not start gamerunner!");

        let bpath = match self.book_path {
            OpeningBook::Epd(path) => path,
            OpeningBook::Pgn(path) => path,
        };

        File::open(bpath).expect("Could not find opening book!");

        let out_dir = self.out_dir;

        fs::create_dir(out_dir).expect("The output directory already exists!");

        fs::create_dir(format!("{out_dir}/nets")).expect("Something went very wrong!");

        let stats_path = format!("{out_dir}/stats.txt");
        let sched_path = format!("{out_dir}/schedule.txt");

        File::create(stats_path.as_str()).expect("Couldn't create stats file!");
        File::create(sched_path.as_str()).expect("Couldn't create schedule file!");

        let mut sched =
            fs::OpenOptions::new().write(true).open(sched_path.as_str()).expect("Couldn't open schedule file!");
        writeln!(&mut sched, "{schedule:#?}").expect("Couldn't write schedule to file!");

        let base_path_string = format!("{out_dir}/base_engine");
        let dev_path_string = format!("{out_dir}/dev_engine");

        let base_exe_path = format!("{base_path_string}/base_engine");
        let base_engine = &self.base_engine;

        clone(base_engine, base_path_string.as_str());

        println!("# [Building {}/{}]", base_engine.repo, base_engine.branch);
        base_engine.engine_type.build(base_path_string.as_str(), "base_engine", base_engine.net_path).unwrap();

        println!("# [Running Bench]");
        let bench = base_engine.engine_type.bench(&base_exe_path).unwrap();
        if let Some(expected) = base_engine.bench {
            assert_eq!(bench, expected, "Bench did not match!")
        }

        println!("# [Bench Successfull]");

        let dev_engine = &self.dev_engine;

        clone(dev_engine, dev_path_string.as_str());
    }

    pub fn dispatch(&self, net_id: &str, superbatch: usize) -> JoinHandle<()> {
        let out_dir = self.out_dir;

        let name = format!("{net_id}-{superbatch}");
        println!("Testing [{}]", logger::ansi(name.as_str(), 31));

        let dev_path_string = format!("{out_dir}/dev_engine");
        let base_engine_path = format!("{out_dir}/base_engine/base_engine");

        let dev_engine_path = format!("{out_dir}/nets/{name}/{name}");

        self.dev_engine
            .engine_type
            .build(
                dev_path_string.as_str(),
                &format!("../nets/{name}/{name}"),
                Some(&format!("../nets/{name}/quantised.bin")),
            )
            .expect("Failed to build dev engine!");

        let _bench = self.dev_engine.engine_type.bench(dev_engine_path.as_str()).expect("Failed to bench dev engine!");

        let (opening_book, is_pgn) = match self.book_path {
            OpeningBook::Epd(path) => (path.to_string(), false),
            OpeningBook::Pgn(path) => (path.to_string(), true),
        };

        let args = GameRunnerArgs {
            gamerunner_path: self.gamerunner_path.as_internal(),
            dev_engine_path,
            base_engine_path,
            dev_options: self.dev_engine.uci_options.iter().map(UciOption::to_string).collect(),
            base_options: self.base_engine.uci_options.iter().map(UciOption::to_string).collect(),
            time_control: self.time_control,
            opening_book,
            is_pgn,
            num_game_pairs: self.num_game_pairs,
            concurrency: self.concurrency,
        };

        let stats_path = format!("{out_dir}/stats.txt");

        thread::spawn(move || {
            let (elo, err) = gamerunner::run_games(args).unwrap();
            let mut file =
                std::fs::OpenOptions::new().append(true).open(stats_path.as_str()).expect("Couldn't open stats path!");

            writeln!(file, "{superbatch}, {elo}, {err}").expect("Couldn't write to file!");
        })
    }
}

fn clone<T: EngineType>(engine: &Engine<T>, out_dir: &str) {
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

pub struct OpenBenchCompliant;
impl EngineType for OpenBenchCompliant {
    fn build(&self, repo_path: &str, out_path: &str, net: Option<&str>) -> Result<(), String> {
        let mut build_base = Command::new("make");

        build_base.current_dir(repo_path).arg(format!("EXE={out_path}"));

        if let Some(net_path) = net {
            build_base.arg(format!("EVALFILE={net_path}"));
        }

        match build_base.output() {
            io::Result::Err(err) => Err(format!("Failed to build engine: {err}!")),
            io::Result::Ok(out) => {
                if out.status.success() {
                    Ok(())
                } else {
                    println!("{}", String::from_utf8(out.stdout).unwrap());
                    Err(String::from("Failed to build engine!"))
                }
            }
        }
    }

    fn bench(&self, path: &str) -> Result<usize, String> {
        let mut bench_cmd = Command::new(path);

        let output = bench_cmd.arg("bench").output().expect("Failed to run bench on engine!");

        assert!(output.status.success(), "Failed to run bench on engine!");

        let out = String::from_utf8(output.stdout).expect("Could not parse bench output!");

        let split = out.split_whitespace();

        let mut bench = None;

        let mut prev = "what";
        for word in split {
            if word == "nodes" {
                bench = prev.parse().ok();
                break;
            }

            prev = word;
        }

        if let Some(bench) = bench {
            Ok(bench)
        } else {
            Err(String::from("Failed to run bench!"))
        }
    }
}
