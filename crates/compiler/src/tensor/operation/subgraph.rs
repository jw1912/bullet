use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::tensor::{IRTrace, NodeId, OpType, TType, TValue, TensorIR};

#[derive(Clone, Debug)]
pub struct SubGraph {
    graph: TensorIR,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl SubGraph {
    pub fn new(graph: TensorIR, inputs: Vec<NodeId>, outputs: Vec<NodeId>) -> Result<Self, IRTrace> {
        if inputs.len() != inputs.iter().collect::<HashSet<_>>().len() {
            return Err("Duplicate inputs to subgraph!".into());
        }

        if outputs.len() != outputs.iter().collect::<HashSet<_>>().len() {
            return Err("Duplicate outputs of subgraph!".into());
        }

        for &node in &inputs {
            graph.get_node(node)?;

            if !graph.is_input(node)? {
                return Err("Not input to subgraph!".into());
            }
        }

        for &node in &outputs {
            graph.get_node(node)?;

            if !graph.is_output(node) {
                return Err("Not output of subgraph!".into());
            }
        }

        Ok(Self { graph, inputs, outputs })
    }

    pub fn internal_graph(&self) -> &TensorIR {
        &self.graph
    }

    pub fn internal_inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    pub fn internal_outputs(&self) -> &[NodeId] {
        &self.outputs
    }
}

impl OpType for SubGraph {
    fn opname(&self) -> String {
        "subgraph".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        self.inputs.iter().map(|i| self.graph.get_node(*i).unwrap().ty()).collect()
    }

    fn outputs(&self) -> Vec<TType> {
        self.outputs.iter().map(|o| self.graph.get_node(*o).unwrap().ty()).collect()
    }

    fn evaluate(&self, inputs: Vec<&TValue>, outputs: Vec<&mut TValue>) -> bool {
        let inputs = inputs.iter().cloned().cloned();
        let inputs: HashMap<NodeId, TValue> = self.inputs.iter().cloned().zip(inputs).collect();

        let Some(outs) = self.graph.evaluate(inputs).unwrap() else { return false };

        for (id, out) in self.outputs.iter().zip(outputs) {
            *out = outs.get(id).unwrap().clone();
        }

        true
    }

    fn equals(&self, _other: &Rc<dyn OpType>) -> bool {
        false
    }
}
