pub mod adam;
pub mod clip;
pub mod decay;
pub mod radam;
pub mod ranger;
pub mod utils;

use std::{collections::BTreeMap, fmt::Debug, marker::PhantomData, sync::Arc};

use bullet_compiler::tensor::TValue;
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    kernel::CompiledKernel,
    runtime::{Gpu, Stream},
};

use crate::model::{Model, TensorMap};

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

type OptimiserUpdateResult<'a, G> = Result<OptimiserUpdateSync<'a, G>, <G as Gpu>::Error>;

pub trait OptimiserState<G: Gpu>: Sized {
    type Params: Clone + Debug + Default;

    fn new(strean: &Arc<Stream<G>>, size: usize, params: Self::Params) -> Result<Self, G::Error>;

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
    pub model: Model<G>,
    pub state: BTreeMap<String, S>,
    pre_update: Vec<Box<dyn AdditionalUpdate<G>>>,
    post_update: Vec<Box<dyn AdditionalUpdate<G>>>,
}

pub trait AdditionalUpdate<G: Gpu>: 'static {
    fn apply_update<'a>(&'a mut self, model: &Model<G>) -> OptimiserUpdateResult<'a, G>;
}

impl<G: Gpu, S: OptimiserState<G>> Optimiser<G, S> {
    pub fn new(model: Model<G>, params: S::Params) -> Result<Self, G::Error> {
        let mut state = BTreeMap::new();
        let stream = model.stream();

        for (id, value) in model.weights() {
            let size = value.size();
            let single = S::new(&stream, size, params.clone())?;
            let old = state.insert(id.clone(), single);
            assert!(old.is_none());
        }

        Ok(Self { model, state, pre_update: Vec::new(), post_update: Vec::new() })
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
        let mut sync = OptimiserUpdateSync::default();

        for additional in &mut self.pre_update {
            sync.extend_by(additional.apply_update(&self.model)?);
        }

        for (id, single) in &mut self.state {
            let weight = self.model.weights().get(id).unwrap();

            if let Some(grads) = gradients.get(id) {
                sync.extend_by(single.update(
                    stream,
                    weight.clone(),
                    grads.clone(),
                    gradient_factor.clone(),
                    learning_rate.clone(),
                )?);
            }
        }

        for additional in &mut self.post_update {
            sync.extend_by(additional.apply_update(&self.model)?);
        }

        Ok(sync)
    }

    pub fn reset_state(&mut self) -> Result<(), G::Error> {
        for single in self.state.values_mut() {
            single.reset()?;
        }

        Ok(())
    }

    pub fn set_params_for_weight(&mut self, id: &str, params: S::Params) {
        self.state.get_mut(id).unwrap().set_params(params).unwrap();
    }

    pub fn set_params(&mut self, params: S::Params) {
        for id in self.model.weights().clone().keys() {
            self.set_params_for_weight(id, params.clone());
        }
    }

    pub fn write_to_checkpoint(&self, path: &str) -> Result<(), G::Error> {
        let mut file = std::fs::File::create(format!("{path}/weights.bin")).unwrap();
        self.model.write_to(&mut file)?;
        let map = self.state.iter().map(|(id, single)| (id.clone(), single)).collect();
        S::write_to_checkpoint(&map, path)
    }

    pub fn load_weights_from_file(&mut self, path: &str) -> Result<(), G::Error> {
        self.model.load_from(std::fs::File::open(path).unwrap())
    }

    pub fn load_from_checkpoint(&mut self, path: &str) -> Result<(), G::Error> {
        self.load_weights_from_file(&format!("{path}/weights.bin"))?;
        let mut map = self.state.iter_mut().map(|(id, single)| (id.clone(), single)).collect();
        S::load_from_checkpoint(&mut map, path)
    }
}

pub struct WrapOptimiser<O, P> {
    optimiser: O,
    phantom_data: PhantomData<P>,
}

impl<G, O, P> OptimiserState<G> for WrapOptimiser<O, P>
where
    G: Gpu,
    O: OptimiserState<G>,
    P: Clone + Default + Debug + Into<O::Params>,
{
    type Params = P;

    fn new(stream: &Arc<Stream<G>>, size: usize, params: Self::Params) -> Result<Self, G::Error> {
        Ok(Self { optimiser: O::new(stream, size, params.into())?, phantom_data: PhantomData })
    }

    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
        self.optimiser.update(stream, weights, grads, gradient_factor, learning_rate)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        self.optimiser.reset()
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        self.optimiser.set_params(params.into())
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.optimiser)).collect();
        O::load_from_checkpoint(&mut map, path)
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.optimiser)).collect();
        O::write_to_checkpoint(&map, path)
    }
}
