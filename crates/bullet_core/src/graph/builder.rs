mod affine;
mod node;

use acyclib::graph::NodeId;
pub use affine::Affine;
pub use node::{Activation, GraphBuilderNode};

use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard},
};

use crate::{
    device::Device,
    graph::{
        Graph, GraphNodeId, GraphNodeIdTy,
        ir::{
            BackendMarker, GraphIRManager,
            operation::{
                GraphIROperationCompilable,
                unary::{Reduce, ReduceAcrossBatch},
            },
        },
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
pub struct GraphBuilder<B: BackendMarker> {
    ir: Mutex<GraphIRManager<B>>,
    init_data: Mutex<HashMap<String, InitSettings>>,
    consts: Mutex<HashMap<NodeId, Vec<f32>>>,
    dump_graphviz: Mutex<Option<String>>,
}

impl<B: BackendMarker> GraphBuilder<B> {
    pub fn ir(&self) -> MutexGuard<'_, GraphIRManager<B>> {
        self.ir.try_lock().unwrap()
    }

    fn init(&self) -> MutexGuard<'_, HashMap<String, InitSettings>> {
        self.init_data.try_lock().unwrap()
    }

    pub fn apply(&self, operation: impl GraphIROperationCompilable<B>) -> GraphBuilderNode<'_, B> {
        match self.ir().add_op(operation) {
            Ok(node) => GraphBuilderNode { node, builder: self },
            Err(e) => {
                println!("{e:#?}");
                panic!();
            }
        }
    }

    pub fn new_dense_input<'a>(&'a self, id: &str, shape: Shape) -> GraphBuilderNode<'a, B> {
        let node = self.ir().add_dense_input(id, shape).unwrap();
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_sparse_input<'a>(&'a self, id: &str, shape: Shape, nnz: usize) -> GraphBuilderNode<'a, B> {
        let node = self.ir().add_sparse_input(id, shape, nnz).unwrap();
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_constant<'a>(&'a self, shape: Shape, vals: &[f32]) -> GraphBuilderNode<'a, B> {
        let node = self.ir().add_constant(shape).unwrap();
        assert_eq!(shape.size(), vals.len(), "Shape of constant does not match provided values!");
        self.consts.try_lock().unwrap().insert(node.idx, vals.to_vec());
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_weights<'a>(&'a self, id: &str, shape: Shape, init: InitSettings) -> GraphBuilderNode<'a, B> {
        let node = self.ir().add_weights(id, shape).unwrap();
        self.init().insert(id.to_string(), init);
        GraphBuilderNode { node, builder: self }
    }

    pub fn new_affine(&self, id: &str, input_size: usize, output_size: usize) -> Affine<'_, B> {
        self.new_affine_custom(id, input_size, output_size, 1)
    }

    pub fn new_affine_custom(
        &self,
        id: &str,
        input_size: usize,
        output_size: usize,
        bias_cols: usize,
    ) -> Affine<'_, B> {
        let wid = format!("{id}w");
        let init = InitSettings::Normal { mean: 0.0, stdev: (2.0 / (input_size as f32 * bias_cols as f32)).sqrt() };
        let weights = self.new_weights(&wid, Shape::new(output_size, input_size), init);
        let bias = self.new_weights(&format!("{id}b"), Shape::new(output_size, bias_cols), InitSettings::Zeroed);

        Affine { weights, bias }
    }

    /// Outputs the `GraphIR` before and after optimisation to the given `path`, at compilation time.
    pub fn dump_graphviz(&self, path: &str) {
        *self.dump_graphviz.try_lock().unwrap() = Some(path.to_string());
    }
}

impl<D: Device<Marker = B>, B: BackendMarker<Backend = D>> GraphBuilder<B> {
    pub fn build(self, device: D) -> Graph<D> {
        let mut ir = self.ir.into_inner().unwrap();
        let root = ir.root().unwrap();

        if ir.get(root.idx).unwrap().ty().batched {
            ir.add_op(ReduceAcrossBatch { input: root, reduction: Reduce::Sum }).unwrap();
        }

        if let Some(path) = self.dump_graphviz.try_lock().unwrap().clone() {
            use std::io::Write;
            let opts = "style=filled;\ncolor=lightgrey;\nnode [style=filled,color=white];\n";
            let unoptim = ir.as_graphviz("unoptim").unwrap();
            let unoptim = format!("subgraph cluster_0 {{\nlabel=\"Unoptimised\";\n{opts}{unoptim}}}");

            ir.optimise().unwrap();

            let optim = ir.as_graphviz("optim").unwrap();
            let optim = format!("subgraph cluster_1 {{\nlabel=\"Optimised\";\n{opts}{optim}}}");

            let mut file = std::fs::File::create(path).unwrap();
            write!(&mut file, "digraph G {{\n{unoptim}\n{optim}}}").unwrap();
        } else {
            ir.optimise().unwrap();
        }

        let graph = ir.compile(device).unwrap();

        for (id, init_data) in self.init_data.lock().unwrap().iter() {
            match *init_data {
                InitSettings::Zeroed => {}
                InitSettings::Normal { mean, stdev } => graph
                    .get_mut(GraphNodeId::new(graph.weight_idx(id).unwrap(), GraphNodeIdTy::Values))
                    .unwrap()
                    .seed_random(mean, stdev, true)
                    .unwrap(),
                InitSettings::Uniform { mean, stdev } => graph
                    .get_mut(GraphNodeId::new(graph.weight_idx(id).unwrap(), GraphNodeIdTy::Values))
                    .unwrap()
                    .seed_random(mean, stdev, false)
                    .unwrap(),
            };
        }

        for (&idx, vals) in self.consts.lock().unwrap().iter() {
            graph
                .get_mut(GraphNodeId::new(idx, GraphNodeIdTy::Values))
                .unwrap()
                .load_dense_from_slice(None, vals)
                .unwrap();
        }

        graph
    }
}
