mod canonicalise;
mod decompose;
mod eliminate;
mod fold_constants;

use crate::{
    common::{Binary, DTypeTensor, Unary},
    elementwise::ElementwiseNode,
    ir::{
        IrError, IrGraph,
        node::{IrNode, IrNodeId, IrType},
        operation::{Constant, IrBinary, IrElementwise, IrOperation, IrOperationId, IrOperationType, IrUnary, Leaf},
    },
};

impl IrGraph {
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
            return Err("IrGraph::add_op: invalid operation outputs!".into());
        }

        Ok(output_ids)
    }

    pub fn remove_op(&mut self, id: IrOperationId) -> Result<(), IrError> {
        fn check(cond: bool, msg: impl Into<String>) -> Result<(), IrError> {
            cond.then_some(()).ok_or(format!("IrGraph::remove_op: {}!", msg.into()).into())
        }

        for output in self.get_op(id)?.outputs().to_vec() {
            let node = self.nodes.remove(&output).expect("Node must be present `nodes`!");

            check(node.children() == 0, "node has children")?;
            check(!self.outputs.contains(&output), "node is required")?;

            let op_id = self.links.remove(&output).expect("Node must be present `links`!");
            check(op_id == id, format!("operation id mismatch ({op_id:?} != {id:?})"))?;
        }

        for input in self.get_op(id)?.inputs().to_vec() {
            self.get_node_mut(input)?.dec_children();
        }

        self.ops.remove(&id).expect("Already verified node is present `ops`!");

        Ok(())
    }

    pub fn swap_outputs(&mut self, id1: IrNodeId, id2: IrNodeId) -> Result<(), IrError> {
        let node1 = self.get_node(id1)?;
        let node2 = self.get_node(id2)?;

        if node1.ty() != node2.ty() {
            return Err(format!("IrGraph::swap_outputs: type mismatch {:?} != {:?}!", node1.ty(), node2.ty()).into());
        }

        let op1 = self.get_parent_op(id1)?;
        let op2 = self.get_parent_op(id2)?;

        *self.links.get_mut(&id1).ok_or("IrGraph::swap_outputs: node does not exist!")? = op2;
        *self.links.get_mut(&id2).ok_or("IrGraph::swap_outputs: node does not exist!")? = op1;

        self.get_op_mut(op1)?.swap_output_with(id2, id1)?;
        self.get_op_mut(op2)?.swap_output_with(id1, id2)?;

        // swapping is an opportunity to introduce cycles...
        let _ = self.topo_order_ops()?;

        Ok(())
    }

    pub fn replace_input(&mut self, new: IrNodeId, old: IrNodeId) -> Result<(), IrError> {
        if self.get_node_type(new)? != self.get_node_type(old)? {
            return Err("IrGraph::replace_input: mismatched types!".into());
        }

        for op_id in self.ops.keys().cloned().collect::<Vec<_>>() {
            let count = self.get_op_mut(op_id)?.swap_input_with(new, old);
            self.get_node_mut(new)?.children += count;
            self.get_node_mut(old)?.children = 0;
        }

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
            return Err("IrGraph::replace_op: new output length doesn't match old".into());
        }

        for (new_out, old_out) in new_outs.into_iter().zip(old_outs) {
            self.swap_outputs(new_out, old_out)?;
        }

        Ok(())
    }

    #[must_use]
    pub fn add_leaf(&mut self, ty: IrType) -> IrNodeId {
        self.add_op([], Leaf(ty)).expect("Constructing leaf is infallible!")[0]
    }

    #[must_use]
    pub fn add_const(&mut self, value: DTypeTensor) -> IrNodeId {
        self.add_op([], Constant(value)).expect("Constructing leaf is infallible!")[0]
    }

    pub fn add_unary(&mut self, node: IrNodeId, op: Unary) -> Result<IrNodeId, IrError> {
        self.add_op([node], IrUnary::new(self.get_node_type(node)?, op)).map(|x| x[0])
    }

    pub fn add_binary(&mut self, lhs: IrNodeId, rhs: IrNodeId, op: Binary) -> Result<IrNodeId, IrError> {
        self.add_op([lhs, rhs], IrBinary::new(self.get_node_type(lhs)?, self.get_node_type(rhs)?, op)?).map(|x| x[0])
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
}

#[cfg(test)]
mod tests {
    use crate::common::DType;

    use super::*;

    #[test]
    fn construct_deconstruct() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::F32));
        let z = ir.add_leaf(IrType::new(8, DType::F32));

        let w = ir.add_binary(x, y, Binary::Add)?;
        let t = ir.add_binary(z, w, Binary::Mul)?;
        let u = ir.add_binary(t, x, Binary::Add)?;

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
        let z = ir.add_binary(x, y, Binary::Add)?;
        let w = ir.add_binary(z, y, Binary::Add)?;
        let t = ir.add_binary(w, y, Binary::Sub)?;

        ir.register_output(t);

        let new_t = ir.add_binary(x, y, Binary::Add)?;
        ir.swap_outputs(new_t, t)?;
        ir.eliminate_unused_ops()?;

        assert_eq!(ir.num_ops(), 3);
        assert_eq!(ir.num_nodes(), 3);
        assert!(ir.get_node(x).is_ok());
        assert!(ir.get_node(y).is_ok());
        assert!(ir.get_node(t).is_ok());

        ir.check_valid()
    }

    #[test]
    fn replace_input() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::F32));
        let z = ir.add_binary(x, y, Binary::Add)?;
        let w = ir.add_binary(z, y, Binary::Add)?;
        let t = ir.add_binary(w, y, Binary::Sub)?;

        ir.register_output(t);

        ir.replace_input(x, w)?;

        ir.eliminate_unused_ops()?;

        assert_eq!(ir.num_ops(), 3);
        assert_eq!(ir.num_nodes(), 3);
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
        let z = ir.add_binary(x, y, Binary::Add)?;
        let w = ir.add_binary(z, y, Binary::Add)?;
        let t = ir.add_binary(w, y, Binary::Sub)?;

        ir.register_output(t);

        let new_op = IrBinary::new(ir.get_node_type(x)?, ir.get_node_type(y)?, Binary::Add)?;
        ir.replace_op(ir.get_parent_op(t)?, [x, y], new_op)?;
        ir.eliminate_unused_ops()?;

        assert_eq!(ir.num_ops(), 3);
        assert_eq!(ir.num_nodes(), 3);
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

        assert!(ir.add_binary(x, y, Binary::Add).is_err());

        Ok(())
    }

    #[test]
    fn invalid_addition_dtype() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::I32));

        assert!(ir.add_binary(x, y, Binary::Add).is_err());

        Ok(())
    }

    #[test]
    fn invalid_removal() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::F32));
        let z = ir.add_binary(x, y, Binary::Add)?;

        assert_eq!(ir.get_node(z)?.ty(), IrType::new(8, DType::F32));
        assert!(ir.remove_op(ir.get_parent_op(y)?).is_err());

        Ok(())
    }

    #[test]
    fn invalid_swap_outputs() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(8, DType::F32));
        let y = ir.add_leaf(IrType::new(8, DType::F32));
        let z = ir.add_binary(x, y, Binary::Add)?;
        let w = ir.add_binary(z, y, Binary::Add)?;
        let t = ir.add_binary(w, y, Binary::Sub)?;

        assert_eq!(ir.swap_outputs(z, t), Err("IrGraph::topo_order_ops: cycle found!".into()));

        Ok(())
    }
}
