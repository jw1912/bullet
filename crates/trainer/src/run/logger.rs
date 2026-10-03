use std::{
    fmt::Display,
    io::{Write, stdout},
    sync::atomic::{AtomicBool, Ordering::SeqCst},
    time::{Duration, Instant},
};

use crate::run::Step;

use super::{TrainingEvent, schedule::TrainingSteps};

static CBCS: AtomicBool = AtomicBool::new(false);

pub struct ConsoleObserver {
    steps: TrainingSteps,
    log_rate: usize,
    previous_learning_rate: Option<f32>,
}

impl ConsoleObserver {
    pub fn new(steps: TrainingSteps, log_rate: usize) -> Self {
        Self { steps, log_rate, previous_learning_rate: None }
    }

    pub fn on_event(&mut self, event: &TrainingEvent) {
        match event {
            TrainingEvent::RunStarted(event) => {
                self.steps = event.steps;
                self.previous_learning_rate = None;

                clear_colours();

                println!(
                    "{}",
                    ansi(
                        format!(
                            "Training on {} ({})",
                            event.device_name,
                            event.device_arch.as_deref().unwrap_or("unknown")
                        ),
                        "34;1"
                    )
                );
            }

            TrainingEvent::BatchCompleted(event) => {
                if event.step.batch() == 0 {
                    if let Some(previous) = self.previous_learning_rate {
                        if event.learning_rate < previous {
                            println!("LR dropped to {}", ansi(event.learning_rate, num_cs()));
                        } else if event.learning_rate > previous {
                            println!("LR increased to {}", ansi(event.learning_rate, num_cs()));
                        }
                    }
                }

                self.previous_learning_rate = Some(event.learning_rate);

                if self.log_rate != 0 && event.step.batch().is_multiple_of(self.log_rate) {
                    report_progress(
                        event.step,
                        event.superbatch_elapsed,
                        (event.step.batch() + 1) * self.steps.batch_size,
                    );
                }
            }

            TrainingEvent::SuperbatchCompleted(event) => {
                report_superbatch_finished(
                    event.step.superbatch(),
                    event.loss,
                    event.superbatch_elapsed.as_secs_f32(),
                    event.elapsed.as_secs_f32(),
                    self.steps.batch_size * self.steps.batches_per_superbatch,
                );

                report_time_left(self.steps, event.step.superbatch(), event.elapsed.as_secs_f32());
            }

            TrainingEvent::RunCompleted(event) => {
                let (hours, minutes, seconds) = seconds_to_hms(event.elapsed.as_secs() as u32);

                println!(
                    "Total Training Time: {}h {}m {}s",
                    ansi(hours, num_cs()),
                    ansi(minutes, num_cs()),
                    ansi(seconds, num_cs()),
                );
            }

            TrainingEvent::TrainingReady(_) => {}
        }
    }
}

pub fn ansi<T: Display, U: Display>(x: T, y: U) -> String {
    format!("\x1b[{y}m{x}\x1b[0m{}", esc())
}

pub fn set_colour<U: Display>(x: U) {
    print!("\x1b[{x}m");
}

pub fn clear_colours() {
    print!("{}", esc());
}

pub fn set_cbcs(val: bool) {
    CBCS.store(val, SeqCst)
}

pub fn num_cs() -> i32 {
    if CBCS.load(SeqCst) { 35 } else { 36 }
}

fn esc() -> &'static str {
    if CBCS.load(SeqCst) { "\x1b[38;5;225m" } else { "" }
}

pub fn report_superbatch_progress(
    step: Step, superbatch_timer: &Instant, superbatch_positions: usize
) {
    report_progress(step, superbatch_timer.elapsed(), superbatch_positions);
}

fn report_progress(step: Step, superbatch_time: Duration, superbatch_positions: usize) {
    let num_cs = num_cs();
    let superbatch_time = superbatch_time.as_secs_f32();
    let completed_batches = step.batch() + 1;
    let pct = completed_batches as f32 / step.batches_per_superbatch() as f32;
    let pos_per_sec = if superbatch_time > 0.0 {
        superbatch_positions as f32 / superbatch_time
    } else {
        0.0
    };

    let seconds = superbatch_time / pct - superbatch_time;

    print!(
        "superbatch {} [{}% ({}/{} batches, {} pos/sec)]\n\
        Estimated time to end of superbatch: {}s     \x1b[F",
        ansi(step.superbatch(), num_cs),
        ansi(format!("{:.1}", pct * 100.0), 35),
        ansi(completed_batches, num_cs),
        ansi(step.batches_per_superbatch(), num_cs),
        ansi(format!("{pos_per_sec:.0}"), num_cs),
        ansi(format!("{seconds:.1}"), num_cs),
    );

    let _ = stdout().flush();
}

pub fn report_superbatch_finished(
    superbatch: usize,
    error: f32,
    superbatch_time: f32,
    total_time: f32,
    positions: usize,
) {
    let num_cs = num_cs();
    let pos_per_sec = positions as f32 / superbatch_time;

    println!(
        "superbatch {} | time {}s | running loss {} | {} pos/sec | total time {}s",
        ansi(superbatch, num_cs),
        ansi(format!("{superbatch_time:.1}"), num_cs),
        ansi(format!("{error:.6}"), num_cs),
        ansi(format!("{pos_per_sec:.0}"), num_cs),
        ansi(format!("{total_time:.1}"), num_cs),
    );
}

pub fn report_superbatch_throughput(superbatch: usize, superbatch_time: f32, total_time: f32, positions: usize) {
    let num_cs = num_cs();
    let pos_per_sec = positions as f32 / superbatch_time;

    println!(
        "superbatch {} | time {}s | {} pos/sec | total time {}s",
        ansi(superbatch, num_cs),
        ansi(format!("{superbatch_time:.1}"), num_cs),
        ansi(format!("{pos_per_sec:.0}"), num_cs),
        ansi(format!("{total_time:.1}"), num_cs),
    );
}

pub fn report_time_left(steps: TrainingSteps, superbatch: usize, total_time: f32) {
    let num_cs = num_cs();
    let finished_superbatches = superbatch - steps.start_superbatch + 1;
    let total_superbatches = steps.end_superbatch - steps.start_superbatch + 1;
    let pct = finished_superbatches as f32 / total_superbatches as f32;
    let time_left = total_time / pct - total_time;

    let (hours, minutes, seconds) = seconds_to_hms(time_left as u32);

    println!(
        "Estimated time remaining in training: {}h {}m {}s",
        ansi(hours, num_cs),
        ansi(minutes, num_cs),
        ansi(seconds, num_cs),
    );
}

pub fn seconds_to_hms(mut seconds: u32) -> (u32, u32, u32) {
    let mut minutes = seconds / 60;
    let hours = minutes / 60;
    seconds -= minutes * 60;
    minutes -= hours * 60;

    (hours, minutes, seconds)
}
