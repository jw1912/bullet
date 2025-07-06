mod affine;
mod node;

pub use affine::Affine;
pub use node::{Activation, GraphBuilderNode};

use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard},
};

use crate::{
    backend::device::Device,
    graph::{
        ir::{
            operation::{
                unary::{Reduce, ReduceAcrossBatch},
                GraphIROperation,
            },
            GraphIR,
        },
        Graph,
    },
};

pub use crate::graph::ir::shape::Shape;

#[derive(Clone, Copy, Debug)]
pub enum InitSettings {
    Zeroed,
    Normal { mean: f32, stdev: f32 },
    Uniform { mean: f32, stdev: f32 },
}

#[derive(Default)]
pub struct GraphBuilder {
    graph_builder: Mutex<GraphIR>,
    init_data: Mutex<HashMap<String, InitSettings>>,
    consts: Mutex<HashMap<usize, Vec<f32>>>,
}

impl GraphBuilder {
    fn builder(&self) -> MutexGuard<GraphIR> {
        self.graph_builder.try_lock().unwrap()
    }

    fn init(&self) -> MutexGuard<HashMap<String, InitSettings>> {
        self.init_data.try_lock().unwrap()
    }

    fn apply(&self, operation: impl GraphIROperation) -> GraphBuilderNode {
        match self.builder().add_op(operation) {
            Ok(node) => GraphBuilderNode { node, builder: self },
            Err(e) => {
                println!("{e:#?}");
                panic!();
            }
        }
    }

    pub fn new_dense_input<'a>(&'a self, id: &str, shape: Shape) -> GraphBuilderNode<'a> {
        let node = self.builder().add_dense_input(id, shape).unwrap();
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_sparse_input<'a>(&'a self, id: &str, shape: Shape, nnz: usize) -> GraphBuilderNode<'a> {
        let node = self.builder().add_sparse_input(id, shape, nnz).unwrap();
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_constant<'a>(&'a self, shape: Shape, vals: &[f32]) -> GraphBuilderNode<'a> {
        let node = self.builder().add_node(None, None, shape, false, false, None).unwrap();
        assert_eq!(shape.size(), vals.len(), "Shape of constant does not match provided values!");
        self.consts.try_lock().unwrap().insert(node.idx, vals.to_vec());
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_weights<'a>(&'a self, id: &str, shape: Shape, init: InitSettings) -> GraphBuilderNode<'a> {
        let node = self.builder().add_weights(id, shape).unwrap();
        self.init().insert(id.to_string(), init);
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_affine(&self, id: &str, input_size: usize, output_size: usize) -> Affine {
        self.new_affine_custom(id, input_size, output_size, 1)
    }

    pub fn new_affine_custom(&self, id: &str, input_size: usize, output_size: usize, bias_cols: usize) -> Affine {
        let wid = format!("{}w", id);
        let init = InitSettings::Normal { mean: 0.0, stdev: (2.0 / (input_size as f32 * bias_cols as f32)).sqrt() };
        let weights = self.new_weights(&wid, Shape::new(output_size, input_size), init);
        let bias = self.new_weights(&format!("{}b", id), Shape::new(output_size, bias_cols), InitSettings::Zeroed);

        Affine { weights, bias }
    }

    pub fn build<D: Device>(self, device: D) -> Graph<D> {
        let mut builder = self.graph_builder.into_inner().unwrap();
        let root = builder.root().unwrap();

        if builder.get(root.idx).unwrap().batched {
            builder.add_op(ReduceAcrossBatch { input: root, reduction: Reduce::Sum }).unwrap();
        }

        let mut graph = builder.compile(device).unwrap();

        for (id, init_data) in self.init_data.lock().unwrap().iter() {
            match *init_data {
                InitSettings::Zeroed => {}
                InitSettings::Normal { mean, stdev } => {
                    graph.get_weights_mut(id).seed_random(mean, stdev, true).unwrap()
                }
                InitSettings::Uniform { mean, stdev } => {
                    graph.get_weights_mut(id).seed_random(mean, stdev, false).unwrap()
                }
            };
        }

        for (&idx, vals) in self.consts.lock().unwrap().iter() {
            graph.get_mut(idx).unwrap().load_dense_from_slice(None, vals).unwrap();
        }

        graph
    }
}
