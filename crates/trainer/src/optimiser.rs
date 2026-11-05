pub mod adam;
pub mod clip;
pub mod decay;
pub mod radam;
pub mod ranger;
pub mod utils;
pub mod wrap;

use std::{
    collections::BTreeMap,
    fmt::Debug,
    fs::File,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use bullet_compiler::tensor::TValue;
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    kernel::CompiledKernel,
    runtime::{Device, Gpu, Stream},
};

use crate::{
    model::{ModelDefinition, ModelWeights, TensorMap},
    parallel::{DataParallel, StepOutput},
    run::{PreparedBatchHost, TrainingError},
};

pub struct OptimiserUpdateSync<'a, G: Gpu> {
    kernels: Vec<SyncOnValue<G, &'a CompiledKernel<G>>>,
    copies: Vec<SyncOnValue<G, &'a TValue>>,
}

impl<'a, G: Gpu> Default for OptimiserUpdateSync<'a, G> {
    fn default() -> Self {
        Self { kernels: Vec::new(), copies: Vec::new() }
    }
}

impl<'a, G: Gpu> OptimiserUpdateSync<'a, G> {
    pub fn push_kernel(&mut self, val: SyncOnValue<G, &'a CompiledKernel<G>>) {
        self.kernels.push(val);
    }

    pub fn push_copy(&mut self, val: SyncOnValue<G, &'a TValue>) {
        self.copies.push(val);
    }

    pub fn extend_by(&mut self, mut other: Self) {
        self.kernels.append(&mut other.kernels);
        self.copies.append(&mut other.copies);
    }

    pub fn sync(self) -> Result<(), G::Error> {
        for kernel in self.kernels {
            kernel.value()?;
        }

        for copy in self.copies {
            copy.value()?;
        }

        Ok(())
    }
}

pub(crate) type OptimiserUpdateResult<'a, G> = Result<OptimiserUpdateSync<'a, G>, <G as Gpu>::Error>;

pub trait OptimiserState<G: Gpu>: Sized {
    type Params: Clone + Debug + Default;

    fn new(device: &Arc<Device<G>>, size: usize, params: Self::Params) -> Result<Self, G::Error>;

    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G>;

    fn reset(&mut self) -> Result<(), G::Error>;

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error>;

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error>;

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error>;
}

pub struct Optimiser<G: Gpu, S: OptimiserState<G>> {
    parallel: DataParallel<G, S>,
    pre_update: Vec<Box<dyn AdditionalUpdate<G>>>,
    post_update: Vec<Box<dyn AdditionalUpdate<G>>>,
    cpu_weights: RwLock<ModelWeights>,
    definition: ModelDefinition,
}

pub trait AdditionalUpdate<G: Gpu>: 'static {
    fn apply_update<'a>(&'a mut self, weights: &TensorMap<G>) -> OptimiserUpdateResult<'a, G>;
}

impl<G: Gpu, S: OptimiserState<G>> Optimiser<G, S> {
    pub fn new(
        definition: ModelDefinition,
        weights: ModelWeights,
        device: Arc<Device<G>>,
        params: S::Params,
    ) -> Result<Self, G::Error> {
        Self::with_devices(definition, weights, &[device], params)
    }

    pub fn with_devices(
        definition: ModelDefinition,
        weights: ModelWeights,
        devices: &[Arc<Device<G>>],
        params: S::Params,
    ) -> Result<Self, G::Error> {
        Ok(Self {
            parallel: DataParallel::new(&definition, &weights, devices, params)?,
            pre_update: Vec::new(),
            post_update: Vec::new(),
            cpu_weights: RwLock::new(weights),
            definition,
        })
    }

    pub fn device(&self) -> Arc<Device<G>> {
        self.parallel.device()
    }

    pub fn weights(&self) -> &TensorMap<G> {
        self.parallel.weights()
    }

    fn read_weights(&'_ self) -> RwLockReadGuard<'_, ModelWeights> {
        self.cpu_weights.read().unwrap()
    }

    fn write_weights(&'_ self) -> RwLockWriteGuard<'_, ModelWeights> {
        self.cpu_weights.write().unwrap()
    }

    pub fn cpu_weights(&'_ self) -> Result<RwLockReadGuard<'_, ModelWeights>, G::Error> {
        self.sync_cpu()?;
        Ok(self.read_weights())
    }

    pub fn definition(&self) -> &ModelDefinition {
        &self.definition
    }

    pub fn add_pre_update(&mut self, additional: impl AdditionalUpdate<G>) {
        self.pre_update.push(Box::new(additional));
    }

    pub fn add_post_update(&mut self, additional: impl AdditionalUpdate<G>) {
        self.post_update.push(Box::new(additional));
    }

    pub fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
        gradients: &TensorMap<G>,
    ) -> OptimiserUpdateResult<'a, G> {
        if self.ranks() > 1 {
            return Err("Optimiser::update only supports a single device!".to_string().into());
        }

        let (pre_update, post_update) = (&mut self.pre_update, &mut self.post_update);
        self.parallel.update(stream, gradient_factor, learning_rate, gradients, pre_update, post_update)
    }

    pub fn reset_state(&mut self) -> Result<(), G::Error> {
        for state in self.parallel.states_mut() {
            for single in state.values_mut() {
                single.reset()?;
            }
        }

        Ok(())
    }

    pub fn set_params_for_weight(&mut self, id: &str, params: S::Params) {
        for state in self.parallel.states_mut() {
            state.get_mut(id).unwrap().set_params(params.clone()).unwrap();
        }
    }

    pub fn set_params(&mut self, params: S::Params) {
        for id in self.parallel.state().keys().cloned().collect::<Vec<_>>() {
            self.set_params_for_weight(&id, params.clone());
        }
    }

    pub fn sync_cpu(&self) -> Result<(), G::Error> {
        self.parallel.read_weights(&mut self.write_weights())
    }

    pub fn write_to_checkpoint(&self, path: &str) -> Result<(), G::Error> {
        self.sync_cpu()?;
        let mut file = File::create(format!("{path}/weights.bin")).unwrap();
        self.read_weights().write_into(&mut file).map_err(|e| format!("{e}"))?;
        self.parallel.write_state(path)
    }

    pub fn load_weights_from_file(&mut self, path: &str) -> Result<(), G::Error> {
        let weights = self.cpu_weights.get_mut().unwrap();
        weights.load_from(File::open(path).unwrap()).map_err(|e| format!("{e}"))?;
        self.parallel.write_weights(weights)
    }

    pub fn load_from_checkpoint(&mut self, path: &str) -> Result<(), G::Error> {
        self.load_weights_from_file(&format!("{path}/weights.bin"))?;
        self.parallel.load_state(path)
    }

    pub(crate) fn ranks(&self) -> usize {
        self.parallel.ranks()
    }

    pub(crate) fn prepare(&mut self, batch_size: usize, grad_norms: bool) -> Result<(), TrainingError<G>> {
        self.parallel.prepare(&self.definition, batch_size, grad_norms)
    }

    pub(crate) fn load_batch(&mut self, batch: &PreparedBatchHost<G>) -> Result<(), G::Error> {
        self.parallel.load_batch(batch)
    }

    pub(crate) fn step(
        &mut self,
        lr: f32,
        next: &mut dyn FnMut() -> Option<PreparedBatchHost<G>>,
    ) -> Result<StepOutput, TrainingError<G>> {
        self.parallel.step(lr, next, &mut self.pre_update, &mut self.post_update)
    }

    pub(crate) fn finish(&mut self) {
        self.parallel.finish();
    }
}
