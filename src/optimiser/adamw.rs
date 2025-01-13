use std::collections::HashMap;

use crate::{nn::Graph, tensor::DenseMatrix};

use super::{utils, Optimiser, OptimiserType};

#[derive(Clone, Copy, Debug)]
pub struct AdamWParams {
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for AdamWParams {
    fn default() -> Self {
        Self { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 }
    }
}

#[derive(Default)]
pub struct AdamW;
impl OptimiserType for AdamW {
    type Optimiser = AdamWOptimiser;
}

pub struct AdamWOptimiser {
    graph: Graph,
    momentum: HashMap<String, DenseMatrix>,
    velocity: HashMap<String, DenseMatrix>,
    params: HashMap<String, AdamWParams>,
}

impl Optimiser for AdamWOptimiser {
    type Params = AdamWParams;

    fn new(graph: Graph, default_params: Self::Params) -> Self {
        let weight_ids = graph.weight_ids();

        let mut momentum = HashMap::new();
        let mut velocity = HashMap::new();
        let mut params = HashMap::new();

        for id in weight_ids {
            let shape = graph.get_weights(&id).values.shape();

            let old = momentum.insert(id.clone(), DenseMatrix::zeroed(shape));
            assert!(old.is_none());

            let old = velocity.insert(id.clone(), DenseMatrix::zeroed(shape));
            assert!(old.is_none());

            let old = params.insert(id, default_params);
            assert!(old.is_none());
        }

        Self { graph, momentum, velocity, params }
    }

    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    fn update(&mut self, gradient_factor: f32, learning_rate: f32) {
        for id in &self.graph.weight_ids() {
            let weights = self.graph.get_weights_mut(id);

            weights.values.dense_mut().adamw(
                weights.gradients.as_ref().unwrap(),
                self.momentum.get_mut(id).unwrap(),
                self.velocity.get_mut(id).unwrap(),
                self.params.get(id).unwrap(),
                gradient_factor,
                learning_rate,
            );
        }
    }

    fn write_to_checkpoint(&self, path: &str) {
        utils::write_graph_weights_to_file(&self.graph, &format!("{path}/weights.bin"));
        utils::write_weight_hashmap_to_file(&self.momentum, &format!("{path}/momentum.bin"));
        utils::write_weight_hashmap_to_file(&self.velocity, &format!("{path}/velocity.bin"));
    }

    fn load_from_checkpoint(&mut self, path: &str) {
        utils::load_graph_weights_from_file(&mut self.graph, &format!("{path}/weights.bin"));
        utils::load_weight_hashmap_from_file(&mut self.momentum, &format!("{path}/momentum.bin"));
        utils::load_weight_hashmap_from_file(&mut self.velocity, &format!("{path}/velocity.bin"));
    }

    fn set_params_for_weight(&mut self, id: &str, params: Self::Params) {
        *self.params.get_mut(id).unwrap() = params;
    }
}

impl AdamWOptimiser {
    pub fn load_weights_from_file(&mut self, path: &str) {
        utils::load_graph_weights_from_file(&mut self.graph, path);
    }
}
