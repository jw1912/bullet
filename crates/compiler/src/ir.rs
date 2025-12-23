pub mod display;
pub mod node;
pub mod operation;
pub mod passes;
pub mod transform;

#[cfg(test)]
mod tests;

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use node::{IrNode, IrNodeId, IrType};
use operation::{IrOperation, IrOperationId, Leaf};

use crate::common::{DType, DTypeTensor, topo_order};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrError(pub String);

impl<T: Into<String>> From<T> for IrError {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

#[derive(Clone, Default, Debug)]
pub struct IrGraph {
    nodes: HashMap<IrNodeId, IrNode>,
    ops: HashMap<IrOperationId, IrOperation>,
    links: HashMap<IrNodeId, IrOperationId>,
    outputs: HashSet<IrNodeId>,
}

impl IrGraph {
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_parent_op(&self, node: IrNodeId) -> Result<IrOperationId, IrError> {
        self.links.get(&node).cloned().ok_or(format!("IrGraph::get_parent_op: node {node:?} does not exist!").into())
    }

    pub fn get_op(&self, op: IrOperationId) -> Result<&IrOperation, IrError> {
        self.ops.get(&op).ok_or(format!("IrGraph::get_op: operation {op:?} does not exist!").into())
    }

    pub fn get_op_mut(&mut self, op: IrOperationId) -> Result<&mut IrOperation, IrError> {
        self.ops.get_mut(&op).ok_or(format!("IrGraph::get_op_mut: operation {op:?} does not exist!").into())
    }

    pub fn get_node(&self, node: IrNodeId) -> Result<&IrNode, IrError> {
        self.nodes.get(&node).ok_or(format!("IrGraph::get_node: node {node:?} does not exist!").into())
    }

    pub fn get_node_mut(&mut self, node: IrNodeId) -> Result<&mut IrNode, IrError> {
        self.nodes.get_mut(&node).ok_or(format!("IrGraph::get_node_mut: node {node:?} does not exist!").into())
    }

    pub fn get_node_type(&self, node: IrNodeId) -> Result<IrType, IrError> {
        self.get_node(node).map(IrNode::ty)
    }

    pub fn topo_order_ops(&self) -> Result<Vec<IrOperationId>, IrError> {
        let edges_rev = self
            .ops
            .iter()
            .map(|(&idx, data)| {
                data.inputs()
                    .iter()
                    .map(|&x| self.get_parent_op(x).map(|x| x.inner()))
                    .collect::<Result<_, _>>()
                    .map(|x| (idx.inner(), x))
            })
            .collect::<Result<_, _>>()?;

        topo_order(edges_rev)
            .ok_or("IrGraph::topo_order_ops: cycle found!".into())
            .map(|x| x.into_iter().map(IrOperationId::from_inner).collect())
    }

    pub fn is_output(&self, node: IrNodeId) -> bool {
        self.outputs.contains(&node)
    }

    pub fn register_output(&mut self, node: IrNodeId) {
        self.outputs.insert(node);
    }

    pub fn unregister_output(&mut self, node: IrNodeId) {
        self.outputs.remove(&node);
    }

    pub fn check_valid(&self) -> Result<(), IrError> {
        let mut registered_outputs = HashSet::new();
        let mut expected_child_count = HashMap::new();
        let mut actual_child_count: HashMap<_, _> = self.nodes.keys().map(|x| (x, 0)).collect();

        fn check<T: Into<String>>(cond: bool, msg: T) -> Result<(), IrError> {
            cond.then_some(()).ok_or(format!("IrGraph::check_valid: {}!", msg.into()).into())
        }

        for op_id in self.topo_order_ops()? {
            let op = self.get_op(op_id)?;

            for input in op.inputs() {
                *actual_child_count.get_mut(input).ok_or("IrGraph::check_valid: unexpected input node!")? += 1;
            }

            let output_types = op.op().outputs();

            check(
                op.outputs().len() == output_types.len(),
                format!(
                    "length of operation outputs ({}) does not match expected ({})",
                    op.outputs().len(),
                    output_types.len()
                ),
            )?;

            for (&output, ty) in op.outputs().iter().zip(output_types) {
                check(registered_outputs.insert(output), "output already registered")?;

                let node = self.get_node(output)?;
                check(node.ty() == ty, format!("output type ({:?}) does not match expected ({ty:?})", node.ty()))?;
                check(
                    node.id() == output,
                    format!("output id ({:?}) does not match expected ({output:?})", node.id()),
                )?;
                check(
                    expected_child_count.insert(output, node.children()).is_none(),
                    "expected child count already present",
                )?;
            }
        }

        for (id, count) in expected_child_count {
            let actual = *actual_child_count.get(&id).ok_or("IrGraph::check_valid: node does not exist!")?;
            check(count == actual, format!("actual child count ({actual}) does not match expected ({count})"))?;
        }

        Ok(())
    }

    pub fn evaluate(
        &self,
        inputs: impl Into<HashMap<IrNodeId, DTypeTensor>>,
    ) -> Result<HashMap<IrNodeId, DTypeTensor>, IrError> {
        let mut values: HashMap<_, _> =
            inputs.into().into_iter().map(|(id, tensor)| (id, RefCell::new(tensor))).collect();

        let mut vars = HashSet::new();

        for (id, tensor) in &values {
            let op = self.get_op(self.get_parent_op(*id)?)?;
            if IrOperation::downcast::<Leaf>(op.op()).is_none() {
                return Err("Seeded non-leaf node!".into());
            }

            let concrete_size = tensor.borrow().size();
            let size = self.get_node_type(*id)?.size();

            if let Some(var) = size.get_var_size(concrete_size) {
                vars.insert(var);
            }
        }

        let var = match vars.len() {
            0 => 1,
            1 => *vars.iter().next().unwrap(),
            _ => return Err(format!("Mismatching batch sizes in inputs: {vars:?}").into()),
        };

        for id in self.topo_order_ops()? {
            let op = self.get_op(id)?;

            for &output in op.outputs() {
                let ty = self.get_node_type(output)?;
                let size = ty.size().evaluate(var);

                let tensor = match ty.dtype() {
                    DType::F32 => DTypeTensor::F32(vec![0.0; size]),
                    DType::I32 => DTypeTensor::I32(vec![0; size]),
                };

                let is_prev = values.contains_key(&output);
                let is_leaf = IrOperation::downcast::<Leaf>(op.op()).is_some();

                if !is_leaf {
                    assert!(values.insert(output, RefCell::new(tensor)).is_none(), "Cannot happen!");
                } else if !is_prev {
                    return Err("Leaf node not seeded!".into());
                }
            }

            let op_inputs = op
                .inputs()
                .iter()
                .map(|i| values.get(i).map(|i| i.borrow()))
                .collect::<Option<Vec<_>>>()
                .ok_or("IrGraph::evaluate: input missing!")?;

            let mut op_outputs = op
                .outputs()
                .iter()
                .map(|i| values.get(i).map(|i| i.borrow_mut()))
                .collect::<Option<Vec<_>>>()
                .ok_or("IrGraph::evaluate: output missing!")?;

            op.op().evaluate(
                &op_inputs.iter().map(|x| &**x).collect::<Vec<_>>(),
                &mut op_outputs.iter_mut().map(|x| &mut **x).collect::<Vec<_>>(),
            );
        }

        Ok(values.into_iter().filter_map(|x| self.outputs.contains(&x.0).then(|| (x.0, x.1.into_inner()))).collect())
    }
}
