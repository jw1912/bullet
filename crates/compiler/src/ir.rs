pub mod node;
pub mod operation;

#[cfg(test)]
mod tests;

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::{self, Debug},
};

use node::{IrNodeId, IrType};
use operation::{IrOperation, IrOperationId, IrOperationType};

use crate::{
    common::{DType, DTypeTensor, topo_order},
    elementwise::{Binary, ElementwiseNode, Unary},
    ir::{node::IrNode, operation::IrElementwise},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrError {
    OpDoesNotExist,
    OpIsNotRoot,
    NodeDoesNotExist,
    NodeIsRequired,
    FailedTypeCheck,
    InvalidOperationInputs,
    InvalidOperationOutputs,
    Cyclic,
    Message(String),
}

impl From<String> for IrError {
    fn from(value: String) -> Self {
        Self::Message(value)
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
        self.links.get(&node).cloned().ok_or(IrError::NodeDoesNotExist)
    }

    pub fn get_op(&self, op: IrOperationId) -> Result<&IrOperation, IrError> {
        self.ops.get(&op).ok_or(IrError::OpDoesNotExist)
    }

    pub fn get_op_mut(&mut self, op: IrOperationId) -> Result<&mut IrOperation, IrError> {
        self.ops.get_mut(&op).ok_or(IrError::NodeDoesNotExist)
    }

    pub fn get_node(&self, node: IrNodeId) -> Result<&IrNode, IrError> {
        self.nodes.get(&node).ok_or(IrError::NodeDoesNotExist)
    }

    pub fn get_node_mut(&mut self, node: IrNodeId) -> Result<&mut IrNode, IrError> {
        self.nodes.get_mut(&node).ok_or(IrError::NodeDoesNotExist)
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

        topo_order(edges_rev).ok_or(IrError::Cyclic).map(|x| x.into_iter().map(IrOperationId::from_inner).collect())
    }

    pub fn add_op(
        &mut self,
        inputs: impl AsRef<[IrNodeId]>,
        op: impl IrOperationType,
    ) -> Result<Vec<IrNodeId>, IrError> {
        let output_ids = (0..op.outputs().len()).map(|_| IrNodeId::default()).collect::<Vec<_>>();
        let output_tys = op.outputs();

        let mut error = false;

        for (&out_id, &out_ty) in output_ids.iter().zip(output_tys.iter()) {
            error |= self.nodes.insert(out_id, IrNode::new(out_id, out_ty)).is_some();
        }

        let inputs = inputs.as_ref().iter().map(|&id| self.get_node(id)).collect::<Result<_, _>>()?;
        let outputs = output_ids.iter().map(|&id| self.get_node(id)).collect::<Result<_, _>>()?;
        let op = IrOperation::new(inputs, outputs, op)?;
        let op_id = op.id();

        for &input in op.inputs() {
            self.get_node_mut(input)?.inc_children();
        }

        error |= self.ops.insert(op_id, op).is_some();

        for &out_id in &output_ids {
            error |= self.links.insert(out_id, op_id).is_some();
        }

        if error {
            return Err(IrError::InvalidOperationOutputs);
        }

        Ok(output_ids)
    }

    pub fn remove_op(&mut self, id: IrOperationId) -> Result<(), IrError> {
        for output in self.get_op(id)?.outputs().to_vec() {
            let node = self.nodes.remove(&output).expect("Node must be present `nodes`!");
            if node.children() > 0 {
                return Err(IrError::OpIsNotRoot);
            }

            if self.outputs.contains(&output) {
                return Err(IrError::NodeIsRequired);
            }

            let op_id = self.links.remove(&output).expect("Node must be present `links`!");
            if op_id != id {
                return Err(IrError::InvalidOperationOutputs);
            }
        }

        for input in self.get_op(id)?.inputs().to_vec() {
            self.get_node_mut(input)?.dec_children();
        }

        self.ops.remove(&id).expect("Already verified node is present `ops`!");

        Ok(())
    }

    #[must_use]
    pub fn add_leaf(&mut self, ty: IrType) -> IrNodeId {
        self.add_op([], operation::Leaf(ty)).expect("Constructing leaf is infallible!")[0]
    }

    pub fn add_unary(&mut self, node: IrNodeId, op: Unary) -> Result<IrNodeId, IrError> {
        self.add_op([node], IrElementwise::unary(self.get_node_type(node)?, op)?).map(|x| x[0])
    }

    pub fn add_binary(&mut self, lhs: IrNodeId, rhs: IrNodeId, op: Binary) -> Result<IrNodeId, IrError> {
        self.add_op([lhs, rhs], IrElementwise::binary(self.get_node_type(lhs)?, self.get_node_type(rhs)?, op)?)
            .map(|x| x[0])
    }

    pub fn add_elementwise<const M: usize, const N: usize, F>(
        &mut self,
        inputs: [IrNodeId; M],
        f: F,
    ) -> Result<[IrNodeId; N], IrError>
    where
        F: for<'a> Fn([ElementwiseNode<'a>; M]) -> Option<[ElementwiseNode<'a>; N]>,
    {
        let nodes = inputs.map(|x| self.get_node_type(x).unwrap());
        let op = IrElementwise::new(nodes, f)?;

        let outs = self.add_op(inputs, op)?;

        let mut output = [outs[0]; N];

        for (i, j) in output.iter_mut().zip(outs) {
            *i = j;
        }

        Ok(output)
    }

    pub fn register_output(&mut self, node: IrNodeId) {
        self.outputs.insert(node);
    }

    pub fn unregister_output(&mut self, node: IrNodeId) {
        self.outputs.remove(&node);
    }

    pub fn swap_outputs(&mut self, id1: IrNodeId, id2: IrNodeId) -> Result<(), IrError> {
        let node1 = self.get_node(id1)?;
        let node2 = self.get_node(id2)?;

        if node1.ty() != node2.ty() {
            return Err(IrError::FailedTypeCheck);
        }

        let op1 = self.get_parent_op(id1)?;
        let op2 = self.get_parent_op(id2)?;

        *self.links.get_mut(&id1).ok_or(IrError::NodeDoesNotExist)? = op2;
        *self.links.get_mut(&id2).ok_or(IrError::NodeDoesNotExist)? = op1;

        self.get_op_mut(op1)?.swap_output_with(id2, id1)?;
        self.get_op_mut(op2)?.swap_output_with(id1, id2)?;

        // swapping is an opportunity to introduce cycles...
        let _ = self.topo_order_ops()?;

        Ok(())
    }

    pub fn replace_op(
        &mut self,
        target: IrOperationId,
        inputs: impl AsRef<[IrNodeId]>,
        op: impl IrOperationType,
    ) -> Result<(), IrError> {
        let new_outs = self.add_op(inputs, op)?;
        let old_outs = self.get_op(target)?.outputs().to_vec();

        if new_outs.len() != old_outs.len() {
            return Err(IrError::InvalidOperationOutputs);
        }

        for (new_out, old_out) in new_outs.into_iter().zip(old_outs) {
            self.swap_outputs(new_out, old_out)?;
        }

        Ok(())
    }

    pub fn eliminate_dead_ops(&mut self) -> Result<(), IrError> {
        for op_id in self.topo_order_ops()?.into_iter().rev() {
            let dead_op = self.get_op(op_id)?.outputs().iter().all(|output| {
                let node = self.get_node(*output).unwrap();
                node.children() == 0 && !self.outputs.contains(output)
            });

            if dead_op {
                self.remove_op(op_id)?;
            }
        }

        Ok(())
    }

    pub fn check_valid(&self) -> Result<(), IrError> {
        let mut registered_outputs = HashSet::new();
        let mut expected_child_count = HashMap::new();
        let mut actual_child_count: HashMap<_, _> = self.nodes.keys().map(|x| (x, 0)).collect();

        for op_id in self.topo_order_ops()? {
            let op = self.get_op(op_id)?;

            for input in op.inputs() {
                *actual_child_count.get_mut(input).ok_or(IrError::NodeDoesNotExist)? += 1;
            }

            let output_types = op.op().outputs();

            if op.outputs().len() != output_types.len() {
                return Err(IrError::InvalidOperationOutputs);
            }

            for (&output, ty) in op.outputs().iter().zip(output_types) {
                if !registered_outputs.insert(output) {
                    return Err(IrError::InvalidOperationOutputs);
                }

                let node = self.get_node(output)?;
                if node.ty() != ty || node.id() != output {
                    return Err(IrError::InvalidOperationOutputs);
                }

                if expected_child_count.insert(output, node.children()).is_some() {
                    return Err(IrError::InvalidOperationOutputs);
                }
            }
        }

        for (id, count) in expected_child_count {
            if count != *actual_child_count.get(&id).ok_or(IrError::NodeDoesNotExist)? {
                return Err(IrError::InvalidOperationOutputs);
            }
        }

        Ok(())
    }

    pub fn evaluate(
        &self,
        inputs: impl Into<HashMap<IrNodeId, DTypeTensor>>,
    ) -> Result<HashMap<IrNodeId, DTypeTensor>, IrError> {
        let mut values: HashMap<_, _> =
            inputs.into().into_iter().map(|(id, tensor)| (id, RefCell::new(tensor))).collect();

        let inputs = values.keys().cloned().collect::<HashSet<_>>();

        let mut vars = HashSet::new();

        for (id, tensor) in &values {
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

                if inputs.contains(&output) {
                    if !values.contains_key(&output) {
                        return Err("Leaf node not seeded!".to_string().into());
                    }
                } else if values.insert(output, RefCell::new(tensor)).is_some() {
                    return Err(IrError::InvalidOperationOutputs);
                }
            }

            let op_inputs = op
                .inputs()
                .iter()
                .map(|i| values.get(i).map(|i| i.borrow()))
                .collect::<Option<Vec<_>>>()
                .ok_or(IrError::InvalidOperationInputs)?;

            let mut op_outputs = op
                .outputs()
                .iter()
                .map(|i| values.get(i).map(|i| i.borrow_mut()))
                .collect::<Option<Vec<_>>>()
                .ok_or(IrError::InvalidOperationOutputs)?;

            op.op().evaluate(
                &op_inputs.iter().map(|x| &**x).collect::<Vec<_>>(),
                &mut op_outputs.iter_mut().map(|x| &mut **x).collect::<Vec<_>>(),
            );
        }

        Ok(values.into_iter().filter_map(|x| self.outputs.contains(&x.0).then(|| (x.0, x.1.into_inner()))).collect())
    }
}

impl fmt::Display for IrGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn map<T>(x: Result<T, IrError>) -> Result<T, fmt::Error> {
            x.map_err(|_| fmt::Error)
        }

        writeln!(f, "start")?;

        for id in map(self.topo_order_ops())? {
            let op = map(self.get_op(id))?;
            let inputs = op.inputs();
            let outputs = op.outputs();

            if outputs.len() > 1 {
                write!(f, "[")?;
            }

            let output_tys =
                map(outputs.iter().map(|x| self.get_node(*x).map(IrNode::ty)).collect::<Result<Vec<_>, _>>())?;

            for (i, (&output, ty)) in outputs.iter().zip(output_tys).enumerate() {
                write!(f, "{output:?} : {ty:?}")?;
                if i != outputs.len() - 1 {
                    write!(f, ", ")?;
                }
            }

            if outputs.len() > 1 {
                write!(f, "]")?;
            }

            write!(f, " = {}(", op.op().opname())?;

            for (i, &input) in inputs.iter().enumerate() {
                write!(f, "{input:?}")?;
                if i != inputs.len() - 1 {
                    write!(f, ", ")?;
                }
            }

            writeln!(f, ")")?;
        }

        write!(f, "return(")?;
        for (i, &output) in self.outputs.iter().enumerate() {
            write!(f, "{output:?}")?;
            if i != self.outputs.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")?;

        Ok(())
    }
}
