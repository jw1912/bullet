use std::collections::{HashMap, HashSet};

use crate::{device::Device, graph::Graph, tensor::DenseMatrix};

use super::{utils, OptimiserState};

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

pub struct AdamW<D: Device> {
    ids: HashSet<String>,
    momentum: HashMap<String, DenseMatrix<D>>,
    velocity: HashMap<String, DenseMatrix<D>>,
    params: HashMap<String, AdamWParams>,
}

impl<D: Device> OptimiserState<D> for AdamW<D> {
    type Params = AdamWParams;

    fn new(graph: &Graph<D>, default_params: Self::Params) -> Self {
        let weight_ids = graph.weight_ids();

        let mut momentum = HashMap::new();
        let mut velocity = HashMap::new();
        let mut params = HashMap::new();

        for id in &weight_ids {
            let w = graph.get_weights(id);
            assert!(w.values.batch_size().is_none());
            let size = w.values.size();

            let old = momentum.insert(id.clone(), DenseMatrix::zeroed(graph.device(), size));
            assert!(old.is_none());

            let old = velocity.insert(id.clone(), DenseMatrix::zeroed(graph.device(), size));
            assert!(old.is_none());

            let old = params.insert(id.clone(), default_params);
            assert!(old.is_none());
        }

        Self { ids: weight_ids.into_iter().collect(), momentum, velocity, params }
    }

    fn update_single_weight(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        id: &str,
        gradient_factor: f32,
        learning_rate: f32,
    ) {
        let params = self.params.get(id).unwrap();
        let momentum = self.momentum.get_mut(id).unwrap();
        let velocity = self.velocity.get_mut(id).unwrap();

        assert!(weights.batch_size().is_none());
        assert!(momentum.batch_size().is_none());
        assert!(velocity.batch_size().is_none());
        assert_eq!(weights.size(), momentum.size());
        assert_eq!(weights.size(), velocity.size());

        D::adamw(
            weights.size(),
            &mut weights.buf,
            &grads.buf,
            &mut momentum.buf,
            &mut velocity.buf,
            params.beta1,
            params.beta2,
            params.min_weight,
            params.max_weight,
            params.decay,
            gradient_factor,
            learning_rate,
        );
    }

    fn reset(&mut self) {
        for id in self.ids.iter() {
            self.momentum.get_mut(id).unwrap().set_zero();
            self.velocity.get_mut(id).unwrap().set_zero();
        }
    }

    fn write_to_checkpoint(&self, path: &str) {
        utils::write_weight_hashmap_to_file(&self.momentum, &format!("{path}/momentum.bin"));
        utils::write_weight_hashmap_to_file(&self.velocity, &format!("{path}/velocity.bin"));
    }

    fn load_from_checkpoint(&mut self, path: &str, old_format: bool) {
        let paths = [format!("{path}/momentum.bin"), format!("{path}/velocity.bin")];
        utils::load_weight_hashmap_from_file(&mut self.momentum, &paths[0], old_format);
        utils::load_weight_hashmap_from_file(&mut self.velocity, &paths[1], old_format);
    }

    fn set_params_for_weight(&mut self, id: &str, params: Self::Params) {
        *self.params.get_mut(id).unwrap() = params;
    }
}
