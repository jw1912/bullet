use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::{
    device::Device,
    graph::{
        builder::Shape,
        instruction::{self, Set},
        ir::{
            node::{GraphIRNode, NodeInfo},
            passes, BackendMarker, GraphIR, GraphIRError, GraphIRNodeInfo,
        },
        tensor::Tensor,
        Graph, GraphFunction, NodeId, NodeIdTy,
    },
};

impl<B: BackendMarker> GraphIR<B>
where
    B::Backend: Device,
{
    pub fn optimise(&mut self) -> Result<(), GraphIRError> {
        while self.try_fusion_pass()? {}

        Ok(())
    }

    pub fn try_fusion_pass(&mut self) -> Result<bool, GraphIRError> {
        for node in self.topo_order()? {
            if self.get(node).is_ok() {
                if let Some(mut transform) = passes::search_for_fusion(self, node)? {
                    transform.eliminated.push(node);
                    self.apply_transform(transform)?;
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    pub fn compile(mut self, device: B::Backend) -> Result<Graph<B::Backend>, GraphIRError> {
        self.is_valid()?;

        if let Some(path) = self.opts.dump_graphviz.clone() {
            use std::io::Write;
            let opts = "style=filled;\ncolor=lightgrey;\nnode [style=filled,color=white];\n";
            let unoptim = self.as_graphviz("unoptim").unwrap();
            let unoptim = format!("subgraph cluster_0 {{\nlabel=\"Unoptimised\";\n{opts}{unoptim}}}");

            self.optimise()?;

            let optim = self.as_graphviz("optim").unwrap();
            let optim = format!("subgraph cluster_1 {{\nlabel=\"Optimised\";\n{opts}{optim}}}");

            let mut file = std::fs::File::create(path).unwrap();
            write!(&mut file, "digraph G {{\n{unoptim}\n{optim}}}").unwrap();
        } else {
            self.optimise()?;
        }

        self.is_valid()?;

        let root = self.root()?.idx;
        let root_data = self.get(root).unwrap().info;

        if !root_data.requires_grad || root_data.batched || root_data.shape != Shape::new(1, 1) {
            return Err(GraphIRError::InvalidRootNode);
        }

        // populate ancillary buffers
        let mut ancillary_buffers = HashMap::new();

        for node in self.nodes.iter() {
            if let Some(op) = &node.1.parent_operation {
                ancillary_buffers.insert(node.0, op.ancillary_buffers(&self)?);
            }
        }

        let device = Arc::new(device);

        let mut nodes = HashMap::new();
        let mut forward = GraphFunction::default();
        let mut backward = GraphFunction::default();

        let mut zero_grads = GraphFunction::default();

        let id_idx_pair = |&node| self.get(node).ok().map(|data| (data.id.clone().unwrap(), node));
        let inputs = self.inputs.iter().filter_map(id_idx_pair).collect();
        let weights = self.weights.iter().filter_map(id_idx_pair).collect();

        let node_info =
            GraphIRNodeInfo { nodes: self.nodes.iter().map(|(idx, GraphIRNode { info, .. })| (*idx, *info)).collect() };

        let topo = self.topo_order()?;

        for GraphIRNode { idx, info, parent_operation, .. } in topo.into_iter().map(|idx| self.get(idx).unwrap()) {
            let idx = *idx;
            let NodeInfo { shape, sparse, requires_grad, .. } = *info;

            let values = Tensor::new(device.clone(), shape, sparse).map_err(|_| GraphIRError::FailedToInitTensor)?;

            nodes.insert(NodeId::new(idx, NodeIdTy::Values), RefCell::new(values));

            if requires_grad {
                let grads = Tensor::new(device.clone(), shape, sparse).map_err(|_| GraphIRError::FailedToInitTensor)?;

                let id = NodeId::new(idx, NodeIdTy::Gradients);
                nodes.insert(id, RefCell::new(grads));

                zero_grads.push(Set(id, 0.0));
            }

            if let Some(op) = parent_operation {
                for (num, &(shape, sparse)) in ancillary_buffers.get(&idx).unwrap().iter().enumerate() {
                    let ancillary =
                        Tensor::new(device.clone(), shape, sparse).map_err(|_| GraphIRError::FailedToInitTensor)?;

                    let id = NodeId::new(idx, NodeIdTy::Ancillary(num as u16));
                    nodes.insert(id, RefCell::new(ancillary));
                }

                forward.extend(op.forward_pass(&node_info, idx));

                if requires_grad {
                    let mut this_bwd = op.backward_pass(&node_info, idx);
                    this_bwd.extend(backward);
                    backward = this_bwd;
                }
            }
        }

        let mut new_bwd = GraphFunction::default();
        new_bwd.push(instruction::Set(NodeId { id: root, ty: NodeIdTy::Gradients }, 1.0));
        new_bwd.extend(backward);
        backward = new_bwd;

        let functions = [("forward", forward), ("backward", backward), ("zero_grads", zero_grads)]
            .into_iter()
            .map(|(x, y)| (x.to_string(), y))
            .collect();

        Ok(Graph { nodes, inputs, weights, functions, device, root, profiles: HashMap::new() })
    }
}
