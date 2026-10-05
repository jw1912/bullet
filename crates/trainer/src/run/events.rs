use std::time::Duration;

use super::{Step, TrainingSteps};

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum TrainingEvent {
    #[non_exhaustive]
    RunStarted { steps: TrainingSteps, device_name: String, device_arch: Option<String> },

    #[non_exhaustive]
    TrainingReady { setup_time: Duration },

    #[non_exhaustive]
    BatchCompleted { step: Step, loss: f32, learning_rate: f32, elapsed: Duration, superbatch_elapsed: Duration },

    #[non_exhaustive]
    SuperbatchCompleted { step: Step, loss: f32, elapsed: Duration, superbatch_elapsed: Duration },

    #[non_exhaustive]
    RunCompleted { elapsed: Duration },
}
