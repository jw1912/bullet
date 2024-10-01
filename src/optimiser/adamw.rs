use std::collections::HashMap;

use crate::{tensor::DenseMatrix, Graph};

use super::{utils, Optimiser};

#[derive(Clone, Copy, Debug)]
pub struct AdamWParams {
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub min_weight: f32,
    pub max_weight: f32,
}

pub struct AdamW {
    graph: Graph,
    momentum: HashMap<String, DenseMatrix>,
    velocity: HashMap<String, DenseMatrix>,
    params: AdamWParams,
}

impl Optimiser for AdamW {
    type Params = AdamWParams;

    fn new(graph: Graph, params: Self::Params) -> Self {
        let weight_ids = graph.weight_ids();

        let mut momentum = HashMap::new();
        let mut velocity = HashMap::new();

        for id in weight_ids {
            let shape = graph.get_weights(&id).values.shape();

            let old = momentum.insert(id.clone(), DenseMatrix::zeroed(shape));
            assert!(old.is_none());

            let old = velocity.insert(id, DenseMatrix::zeroed(shape));
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

            weights.values.adamw(
                weights.gradients.as_mut().unwrap(),
                self.momentum.get_mut(id).unwrap(),
                self.velocity.get_mut(id).unwrap(),
                &self.params,
                gradient_factor,
                learning_rate,
            );
        }
    }

    fn write_to_checkpoint(&self, path: &str) {
        utils::write_graph_weights_component_to_file(&self.graph, &format!("{path}/weights.bin"), false);
        utils::write_graph_weights_component_to_file(&self.graph, &format!("{path}/gradient.bin"), true);
        utils::write_weight_hashmap_to_file(&self.momentum, &format!("{path}/momentum.bin"));
        utils::write_weight_hashmap_to_file(&self.velocity, &format!("{path}/velocity.bin"));
    }

    fn load_from_checkpoint(&mut self, path: &str) {
        utils::load_graph_weights_component_from_file(&mut self.graph, &format!("{path}/weights.bin"), false);
        utils::load_graph_weights_component_from_file(&mut self.graph, &format!("{path}/gradient.bin"), true);
        utils::load_weight_hashmap_from_file(&mut self.momentum, &format!("{path}/momentum.bin"));
        utils::load_weight_hashmap_from_file(&mut self.velocity, &format!("{path}/velocity.bin"));
    }
}
