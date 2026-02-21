pub mod adam;
pub mod clip;
pub mod decay;
pub mod radam;
pub mod ranger;
pub mod utils;

use std::{collections::HashMap, fmt::Debug, marker::PhantomData, sync::Arc};

use crate::{
    model::{Model, TensorMap},
    runtime::{BlockOnDrop, Buffer, Device},
};

pub type OptimiserUpdateValue<D> = Vec<BlockOnDrop<<D as Device>::Stream, Vec<Arc<<D as Device>::Buffer>>>>;

pub trait OptimiserState<D: Device>: Sized {
    type Params: Clone + Debug + Default;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::Error>;

    fn update(
        &mut self,
        stream: &Arc<D::Stream>,
        weights: Arc<D::Buffer>,
        grads: Arc<D::Buffer>,
        gradient_factor: Arc<D::Buffer>,
        learning_rate: Arc<D::Buffer>,
    ) -> Result<OptimiserUpdateValue<D>, D::Error>;

    fn reset(&mut self) -> Result<(), D::Error>;

    fn load_from_checkpoint(map: &mut HashMap<String, &mut Self>, path: &str) -> Result<(), D::Error>;

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::Error>;

    fn set_params(&mut self, params: Self::Params) -> Result<(), D::Error>;
}

pub struct Optimiser<D: Device, S: OptimiserState<D>> {
    phantom: PhantomData<D>,
    pub model: Model<D>,
    pub state: HashMap<String, S>,
    pre_update: Vec<Box<dyn AdditionalUpdate<D>>>,
    post_update: Vec<Box<dyn AdditionalUpdate<D>>>,
}

pub trait AdditionalUpdate<D: Device> {
    fn apply_update(&mut self, model: &Model<D>) -> Result<OptimiserUpdateValue<D>, D::Error>;
}

impl<D: Device, S: OptimiserState<D>> Optimiser<D, S> {
    pub fn new(model: Model<D>, params: S::Params) -> Result<Self, D::Error> {
        let mut state = HashMap::new();

        for (id, value) in model.weights() {
            let size = value.size();
            let single = S::new(model.device(), size, params.clone())?;
            let old = state.insert(id.clone(), single);
            assert!(old.is_none());
        }

        Ok(Self { phantom: PhantomData, model, state, pre_update: Vec::new(), post_update: Vec::new() })
    }

    pub fn add_pre_update(&mut self, additional: impl AdditionalUpdate<D> + 'static) {
        self.pre_update.push(Box::new(additional));
    }

    pub fn add_post_update(&mut self, additional: impl AdditionalUpdate<D> + 'static) {
        self.post_update.push(Box::new(additional));
    }

    pub fn update(
        &mut self,
        stream: &Arc<D::Stream>,
        gradient_factor: Arc<D::Buffer>,
        learning_rate: Arc<D::Buffer>,
        gradients: &TensorMap<D>,
    ) -> Result<Vec<OptimiserUpdateValue<D>>, D::Error> {
        let mut blocks = Vec::new();

        for additional in &mut self.pre_update {
            blocks.push(additional.apply_update(&self.model)?);
        }

        for (id, weight) in self.model.weights() {
            let single = self.state.get_mut(id).unwrap();

            if let Some(grads) = gradients.get(id) {
                blocks.push(single.update(
                    stream,
                    weight.clone(),
                    grads.clone(),
                    gradient_factor.clone(),
                    learning_rate.clone(),
                )?);
            }
        }

        for additional in &mut self.post_update {
            blocks.push(additional.apply_update(&self.model)?);
        }

        Ok(blocks)
    }

    pub fn reset_state(&mut self) -> Result<(), D::Error> {
        for single in self.state.values_mut() {
            single.reset()?;
        }

        Ok(())
    }

    pub fn set_params_for_weight(&mut self, id: &str, params: S::Params) {
        self.state.get_mut(&format!("weights/{id}")).unwrap().set_params(params).unwrap();
    }

    pub fn set_params(&mut self, params: S::Params) {
        for id in self.model.weights().clone().keys() {
            self.set_params_for_weight(id, params.clone());
        }
    }

    pub fn write_to_checkpoint(&self, path: &str) -> Result<(), D::Error> {
        let mut file = std::fs::File::create(format!("{path}/weights.bin")).unwrap();
        self.model.write_to(&mut file)?;
        let map = self.state.iter().map(|(id, single)| (id.clone(), single)).collect();
        S::write_to_checkpoint(&map, path)
    }

    pub fn load_weights_from_file(&mut self, path: &str) -> Result<(), D::Error> {
        self.model.load_from(std::fs::File::open(path).unwrap())
    }

    pub fn load_from_checkpoint(&mut self, path: &str) -> Result<(), D::Error> {
        self.load_weights_from_file(&format!("{path}/weights.bin"))?;
        let mut map = self.state.iter_mut().map(|(id, single)| (id.clone(), single)).collect();
        S::load_from_checkpoint(&mut map, path)
    }
}

pub struct WrapOptimiser<O, P> {
    optimiser: O,
    phantom_data: PhantomData<P>,
}

impl<D, O, P> OptimiserState<D> for WrapOptimiser<O, P>
where
    D: Device,
    O: OptimiserState<D>,
    P: Clone + Default + Debug + Into<O::Params>,
{
    type Params = P;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::Error> {
        Ok(Self { optimiser: O::new(device, size, params.into())?, phantom_data: PhantomData })
    }

    fn update(
        &mut self,
        stream: &Arc<D::Stream>,
        weights: Arc<D::Buffer>,
        grads: Arc<D::Buffer>,
        gradient_factor: Arc<D::Buffer>,
        learning_rate: Arc<D::Buffer>,
    ) -> Result<OptimiserUpdateValue<D>, D::Error> {
        self.optimiser.update(stream, weights, grads, gradient_factor, learning_rate)
    }

    fn reset(&mut self) -> Result<(), D::Error> {
        self.optimiser.reset()
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), D::Error> {
        self.optimiser.set_params(params.into())
    }

    fn load_from_checkpoint(map: &mut HashMap<String, &mut Self>, path: &str) -> Result<(), D::Error> {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.optimiser)).collect();
        O::load_from_checkpoint(&mut map, path)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::Error> {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.optimiser)).collect();
        O::write_to_checkpoint(&map, path)
    }
}
