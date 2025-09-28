pub mod adam;
pub mod clip;
pub mod decay;
pub mod radam;
pub mod ranger;
pub mod utils;

use std::{collections::HashMap, fmt::Debug, marker::PhantomData, sync::Arc};

use crate::{
    device::{Device, OperationError, tensor::DenseMatrix},
    graph::{Graph, GraphNodeId, GraphNodeIdTy, like::GraphLike},
};

pub trait OptimiserState<D: Device>: Sized {
    type Params: Clone + Debug + Default;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError>;

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>>;

    fn reset(&mut self) -> Result<(), D::DeviceError>;

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
    ) -> Result<(), OperationError<D::DeviceError>>;

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError>;

    fn set_params(&mut self, params: Self::Params);
}

pub struct Optimiser<D: Device, G: GraphLike<D>, S: OptimiserState<D>> {
    phantom: PhantomData<D>,
    pub graph: G,
    pub state: HashMap<String, S>,
    pre_update: Vec<Box<dyn AdditionalUpdate<D>>>,
    post_update: Vec<Box<dyn AdditionalUpdate<D>>>,
}

pub trait AdditionalUpdate<D: Device> {
    fn apply_update(&mut self, graph: &mut Graph<D>) -> Result<(), OperationError<D::DeviceError>>;
}

impl<D: Device, G: GraphLike<D>, S: OptimiserState<D>> Optimiser<D, G, S> {
    pub fn new(graph: G, params: S::Params) -> Result<Self, D::DeviceError> {
        let weight_ids = graph.primary().weight_ids();

        let mut state = HashMap::new();

        for id in &weight_ids {
            let idx = graph.primary().weight_idx(id).unwrap();
            let w = graph.primary().get(GraphNodeId::new(idx, GraphNodeIdTy::Values)).unwrap();
            let w = w.dense();
            assert!(w.batch_size().is_none());
            let size = w.size();

            let single = S::new(graph.primary().device(), size, params.clone())?;

            let old = state.insert(id.clone(), single);
            assert!(old.is_none());
        }

        Ok(Self { phantom: PhantomData, graph, state, pre_update: Vec::new(), post_update: Vec::new() })
    }

    pub fn add_pre_update(&mut self, additional: impl AdditionalUpdate<D> + 'static) {
        self.pre_update.push(Box::new(additional));
    }

    pub fn add_post_update(&mut self, additional: impl AdditionalUpdate<D> + 'static) {
        self.post_update.push(Box::new(additional));
    }

    pub fn update(&mut self, gradient_factor: f32, learning_rate: f32) -> Result<(), OperationError<D::DeviceError>> {
        for additional in &mut self.pre_update {
            additional.apply_update(self.graph.primary_mut())?;
        }

        for id in &self.graph.primary().weight_ids() {
            let idx = self.graph.primary().weight_idx(id).unwrap();
            let weight_id = GraphNodeId::new(idx, GraphNodeIdTy::Values);

            let weights = &mut self.graph.primary().get(weight_id)?;
            let single = self.state.get_mut(id).unwrap();

            let grad_id = GraphNodeId::new(idx, GraphNodeIdTy::Gradients);
            if let Ok(grads) = self.graph.primary().get(grad_id) {
                self.graph.reduce_sum_into_first(&self.graph.get_all(grad_id)?)?;
                single.update(&mut *weights.dense_mut(), &mut *grads.dense_mut(), gradient_factor, learning_rate)?;
                self.graph.scatter_first_into_rest(&self.graph.get_all(weight_id)?)?;
            }
        }

        for additional in &mut self.post_update {
            additional.apply_update(self.graph.primary_mut())?;
        }

        Ok(())
    }

    pub fn reset_state(&mut self) -> Result<(), D::DeviceError> {
        for single in self.state.values_mut() {
            single.reset()?;
        }

        Ok(())
    }

    pub fn set_params_for_weight(&mut self, id: &str, params: S::Params) {
        self.state.get_mut(id).unwrap().set_params(params);
    }

    pub fn set_params(&mut self, params: S::Params) {
        for id in self.graph.primary().weight_ids() {
            self.set_params_for_weight(&id, params.clone());
        }
    }

    pub fn write_to_checkpoint(&self, path: &str) -> Result<(), D::DeviceError> {
        self.graph.primary().write_to_file(&format!("{path}/weights.bin"));
        let map = self.state.iter().map(|(id, single)| (id.clone(), single)).collect();
        S::write_to_checkpoint(&map, path)
    }

    pub fn load_from_checkpoint(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_from_checkpoint_(path, false)
    }

    pub fn load_weights_from_file(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_weights_from_file_(path, false)
    }

    pub fn load_from_old_format_checkpoint(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_from_checkpoint_(path, true)
    }

    pub fn load_old_format_weights_from_file(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_weights_from_file_(path, true)
    }

    fn load_weights_from_file_(&mut self, path: &str, old_format: bool) -> Result<(), OperationError<D::DeviceError>> {
        self.graph.primary_mut().load_from_file(path, old_format)?;

        let primary = self.graph.primary();

        for id in primary.weight_ids() {
            let idx = GraphNodeId::new(primary.weight_idx(&id).unwrap(), GraphNodeIdTy::Values);
            self.graph.scatter_first_into_rest(&self.graph.get_all(idx)?)?;
        }

        Ok(())
    }

    fn load_from_checkpoint_(&mut self, path: &str, old_format: bool) -> Result<(), OperationError<D::DeviceError>> {
        self.load_weights_from_file_(&format!("{path}/weights.bin"), old_format)?;
        let mut map = self.state.iter_mut().map(|(id, single)| (id.clone(), single)).collect();
        S::load_from_checkpoint(&mut map, path, old_format)
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

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError> {
        Ok(Self { optimiser: O::new(device, size, params.into())?, phantom_data: PhantomData })
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>> {
        self.optimiser.update(weights, grads, gradient_factor, learning_rate)
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.optimiser.reset()
    }

    fn set_params(&mut self, params: Self::Params) {
        self.optimiser.set_params(params.into());
    }

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
    ) -> Result<(), OperationError<D::DeviceError>> {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.optimiser)).collect();
        O::load_from_checkpoint(&mut map, path, old_format)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError> {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.optimiser)).collect();
        O::write_to_checkpoint(&map, path)
    }
}
