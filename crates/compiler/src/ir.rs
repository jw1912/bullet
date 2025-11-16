pub mod lower;
pub mod node;
pub mod ops;
pub mod size;
pub mod topo;

use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug},
};

use node::{IrNodeId, IrType};
use ops::{IrOp, IrOpId, IrOperation};

use crate::{
    ir::{lower::IrLower, node::IrNode},
    program::{Program, ProgramError},
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
    Lowering(ProgramError),
    Message(String),
}

#[derive(Clone, Default, Debug)]
pub struct IrGraph {
    nodes: HashMap<IrNodeId, IrNode>,
    ops: HashMap<IrOpId, IrOp>,
    links: HashMap<IrNodeId, IrOpId>,
    outputs: HashSet<IrNodeId>,
}

impl IrGraph {
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_parent_op(&self, node: IrNodeId) -> Result<IrOpId, IrError> {
        self.links.get(&node).cloned().ok_or(IrError::NodeDoesNotExist)
    }

    pub fn get_op(&self, op: IrOpId) -> Result<&IrOp, IrError> {
        self.ops.get(&op).ok_or(IrError::OpDoesNotExist)
    }

    pub fn get_op_mut(&mut self, op: IrOpId) -> Result<&mut IrOp, IrError> {
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

    pub fn topo_order_ops(&self) -> Result<Vec<IrOpId>, IrError> {
        let edges_rev = self
            .ops
            .iter()
            .map(|(&idx, data)| {
                data.op()
                    .inputs()
                    .iter()
                    .map(|&x| self.get_parent_op(x).map(|x| x.inner()))
                    .collect::<Result<_, _>>()
                    .map(|x| (idx.inner(), x))
            })
            .collect::<Result<_, _>>()?;

        topo::topo_order(edges_rev).ok_or(IrError::Cyclic).map(|x| x.into_iter().map(IrOpId::from_inner).collect())
    }

    pub fn add_op(&mut self, op: impl IrOperation) -> Result<Vec<IrNodeId>, IrError> {
        let op = IrOp::new(op, self)?;

        let node_tys = op.op().output_types(self)?;
        let node_ids = op.outputs().to_vec();
        let op_id = op.id();

        for &input in op.inputs() {
            self.get_node_mut(input)?.inc_children();
        }

        let mut error = false;

        error |= self.ops.insert(op_id, op).is_some();

        for (&node_id, &node_ty) in node_ids.iter().zip(node_tys.iter()) {
            error |= self.nodes.insert(node_id, IrNode::new(node_id, node_ty)).is_some();
            error |= self.links.insert(node_id, op_id).is_some();
        }

        if error {
            return Err(IrError::InvalidOperationOutputs);
        }

        Ok(node_ids)
    }

    pub fn remove_op(&mut self, id: IrOpId) -> Result<(), IrError> {
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
        self.add_op(ops::Leaf(ty)).expect("Constructing leaf is infallible!")[0]
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

    pub fn replace_op(&mut self, target: IrOpId, op: impl IrOperation) -> Result<(), IrError> {
        let new_outs = self.add_op(op)?;
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

            let output_types = op.op().output_types(self)?;

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

    pub fn lower(&self) -> Result<Program, IrError> {
        let mut lower = IrLower::new(self);

        let topo = self.topo_order_ops()?;

        for op_id in topo {
            let op = self.get_op(op_id)?;
            op.op().lower(&mut lower, op.inputs(), op.outputs())?;
        }

        Ok(lower.finalise())
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

#[cfg(test)]
mod tests {
    use super::{
        node::DType,
        ops::{BinaryOp, MapOp},
        *,
    };

    #[test]
    fn construct_deconstruct() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::F32));
        let z = ir.add_leaf(IrType::new(8, DType::F32));

        let w = ir.add_op(MapOp::Binary { lhs: x, rhs: y, op: BinaryOp::Add })?[0];
        let t = ir.add_op(MapOp::Binary { lhs: z, rhs: w, op: BinaryOp::Mul })?[0];
        let u = ir.add_op(MapOp::Binary { lhs: t, rhs: x, op: BinaryOp::Add })?[0];

        assert_eq!(ir.get_node(u)?.ty(), IrType::new(8, DType::F32));

        ir.remove_op(ir.get_parent_op(u)?)?;
        ir.remove_op(ir.get_parent_op(t)?)?;
        ir.remove_op(ir.get_parent_op(w)?)?;
        ir.remove_op(ir.get_parent_op(z)?)?;
        ir.remove_op(ir.get_parent_op(y)?)?;
        ir.remove_op(ir.get_parent_op(x)?)?;

        assert!(ir.ops.is_empty());
        assert!(ir.nodes.is_empty());
        assert!(ir.links.is_empty());

        ir.check_valid()
    }

    #[test]
    fn swap_outputs() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::F32));
        let z = ir.add_op(MapOp::Binary { lhs: x, rhs: y, op: BinaryOp::Add })?[0];
        let w = ir.add_op(MapOp::Binary { lhs: z, rhs: y, op: BinaryOp::Add })?[0];
        let t = ir.add_op(MapOp::Binary { lhs: w, rhs: y, op: BinaryOp::Sub })?[0];

        ir.register_output(t);

        let new_t = ir.add_op(MapOp::Binary { lhs: x, rhs: y, op: BinaryOp::Add })?[0];
        ir.swap_outputs(new_t, t)?;
        ir.eliminate_dead_ops()?;

        assert_eq!(ir.num_ops(), 3);
        assert_eq!(ir.num_nodes(), 3);
        assert_eq!(ir.get_node(z), Err(IrError::NodeDoesNotExist));
        assert_eq!(ir.get_node(w), Err(IrError::NodeDoesNotExist));
        assert_eq!(ir.get_node(new_t), Err(IrError::NodeDoesNotExist));
        assert!(ir.get_node(x).is_ok());
        assert!(ir.get_node(y).is_ok());
        assert!(ir.get_node(t).is_ok());

        ir.check_valid()
    }

    #[test]
    fn replace_op() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::F32));
        let z = ir.add_op(MapOp::Binary { lhs: x, rhs: y, op: BinaryOp::Add })?[0];
        let w = ir.add_op(MapOp::Binary { lhs: z, rhs: y, op: BinaryOp::Add })?[0];
        let t = ir.add_op(MapOp::Binary { lhs: w, rhs: y, op: BinaryOp::Sub })?[0];

        ir.register_output(t);

        ir.replace_op(ir.get_parent_op(t)?, MapOp::Binary { lhs: x, rhs: y, op: BinaryOp::Add })?;
        ir.eliminate_dead_ops()?;

        assert_eq!(ir.num_ops(), 3);
        assert_eq!(ir.num_nodes(), 3);
        assert_eq!(ir.get_node(z), Err(IrError::NodeDoesNotExist));
        assert_eq!(ir.get_node(w), Err(IrError::NodeDoesNotExist));
        assert!(ir.get_node(x).is_ok());
        assert!(ir.get_node(y).is_ok());
        assert!(ir.get_node(t).is_ok());

        ir.check_valid()
    }

    #[test]
    fn invalid_addition_size() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(16, DType::F32));

        let z = ir.add_op(MapOp::Binary { lhs: x, rhs: y, op: BinaryOp::Add });
        assert_eq!(z, Err(IrError::InvalidOperationInputs), "{z:?}");

        Ok(())
    }

    #[test]
    fn invalid_addition_dtype() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::I32));

        let z = ir.add_op(MapOp::Binary { lhs: x, rhs: y, op: BinaryOp::Add });
        assert_eq!(z, Err(IrError::FailedTypeCheck), "{z:?}");

        Ok(())
    }

    #[test]
    fn invalid_removal() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::F32));
        let z = ir.add_op(MapOp::Binary { lhs: x, rhs: y, op: BinaryOp::Add })?[0];

        assert_eq!(ir.get_node(z)?.ty(), IrType::new(8, DType::F32));
        assert_eq!(ir.remove_op(ir.get_parent_op(y)?), Err(IrError::OpIsNotRoot));

        Ok(())
    }

    #[test]
    fn invalid_swap_outputs() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::F32));
        let z = ir.add_op(MapOp::Binary { lhs: x, rhs: y, op: BinaryOp::Add })?[0];
        let w = ir.add_op(MapOp::Binary { lhs: z, rhs: y, op: BinaryOp::Add })?[0];
        let t = ir.add_op(MapOp::Binary { lhs: w, rhs: y, op: BinaryOp::Sub })?[0];

        assert_eq!(ir.swap_outputs(z, t), Err(IrError::Cyclic));

        Ok(())
    }
}
