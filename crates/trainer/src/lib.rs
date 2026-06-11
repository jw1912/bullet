pub mod model;
pub mod optimiser;
pub mod run;

use bullet_compiler::tensor::IRTrace;
use bullet_gpu::runtime::{self, Device, Gpu};
use optimiser::{Optimiser, OptimiserState};
use run::{
    dataloader::{DataLoader, DataLoadingError},
    schedule::TrainingSchedule,
};

use crate::model::{ModelEvaluator, TensorMap};

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
pub type DefaultDevice = Device<runtime::mock::MockGpu>;

#[cfg(feature = "cuda")]
pub type DefaultDevice = Device<runtime::cuda::Cuda>;

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub type DefaultDevice = Device<runtime::rocm::ROCm>;

#[derive(Debug)]
pub enum TrainerError<G: Gpu> {
    DataLoadingError(DataLoadingError),
    GradientCalculationError(G::Error),
    OptimiserUpdateError(G::Error),
    Unexpected(G::Error),
    CompilingBackwards(IRTrace),
    IoError,
}

impl<G: Gpu> From<DataLoadingError> for TrainerError<G> {
    fn from(value: DataLoadingError) -> Self {
        Self::DataLoadingError(value)
    }
}

pub struct Trainer<G: Gpu, O: OptimiserState<G>, S> {
    pub optimiser: Optimiser<G, O>,
    pub state: S,
    evaluator: Option<ModelEvaluator<G>>,
}

impl<G: Gpu, O: OptimiserState<G>, S> Trainer<G, O, S> {
    pub fn new(optimiser: Optimiser<G, O>, state: S) -> Self {
        Self { optimiser, state, evaluator: None }
    }

    pub fn train_custom(
        &mut self,
        schedule: TrainingSchedule,
        dataloader: impl DataLoader,
        batch_callback: impl FnMut(&mut Self, usize, usize, f32),
        superbatch_callback: impl FnMut(&mut Self, usize),
    ) -> Result<(), TrainerError<G>> {
        run::train_custom(self, schedule, dataloader, batch_callback, superbatch_callback)
    }

    pub fn evaluate(&mut self, inputs: &TensorMap<G>) -> Result<&TensorMap<G>, G::Error> {
        if self.evaluator.is_none() {
            let mut evaluator = ModelEvaluator::new(self.optimiser.definition(), self.optimiser.device())?;
            evaluator.load_device_weights(self.optimiser.weights())?;
            self.evaluator = Some(evaluator);
        }

        self.evaluator.as_mut().unwrap().evaluate(inputs)
    }
}
