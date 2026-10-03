use std::time::Duration;

use super::{Step, TrainingSteps};

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum TrainingEvent {
    RunStarted(RunStarted),
    TrainingReady(TrainingReady),
    BatchCompleted(BatchCompleted),
    SuperbatchCompleted(SuperbatchCompleted),
    RunCompleted(RunCompleted),
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RunStarted {
    pub steps: TrainingSteps,
    pub device_name: String,
    pub device_arch: Option<String>,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TrainingReady {
    pub setup_time: Duration,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct BatchCompleted {
    pub step: Step,
    pub loss: f32,
    pub learning_rate: f32,
    pub completed_batches: usize,
    pub completed_positions: usize,
    pub elapsed: Duration,
    pub superbatch_elapsed: Duration,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SuperbatchCompleted {
    pub step: Step,
    pub loss: f32,
    pub completed_batches: usize,
    pub completed_positions: usize,
    pub elapsed: Duration,
    pub superbatch_elapsed: Duration,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RunCompleted {
    pub completed_batches: usize,
    pub completed_positions: usize,
    pub elapsed: Duration,
}
