mod node;
pub mod operation;

#[cfg(test)]
mod tests;

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, hash_map::Values},
    fmt,
    rc::Rc,
};

use crate::common::{Ansi, Binary, DTypeTensor, Unary, topo_order};

pub use node::{IrNode, IrNodeId, IrType};
use operation::{Constant, IrBinary, IrCopy, IrUnary, Leaf};
pub use operation::{IrOperation, IrOperationId, IrOperationType};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrError(String);

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
        self.links.get(&node).cloned().ok_or(format!("Node {node:?} does not exist!").into())
    }

    pub fn get_op(&self, op: IrOperationId) -> Result<&IrOperation, IrError> {
        self.ops.get(&op).ok_or(format!("Operation {op:?} does not exist!").into())
    }

    pub fn get_op_mut(&mut self, op: IrOperationId) -> Result<&mut IrOperation, IrError> {
        self.ops.get_mut(&op).ok_or(format!("Operation {op:?} does not exist!").into())
    }

    pub fn get_node(&self, node: IrNodeId) -> Result<&IrNode, IrError> {
        self.nodes.get(&node).ok_or(format!("Node {node:?} does not exist!").into())
    }

    pub fn get_node_mut(&mut self, node: IrNodeId) -> Result<&mut IrNode, IrError> {
        self.nodes.get_mut(&node).ok_or(format!("Node {node:?} does not exist!").into())
    }

    pub fn get_node_type(&self, node: IrNodeId) -> Result<IrType, IrError> {
        self.get_node(node).map(IrNode::ty)
    }

    pub fn is_output(&self, node: IrNodeId) -> bool {
        self.outputs.contains(&node)
    }

    pub fn is_input(&self, node: IrNodeId) -> bool {
        let Ok(id) = self.get_parent_op(node) else { return false };
        let Ok(op) = self.get_op(id) else { return false };
        IrOperation::downcast::<Leaf>(op.op()).is_some()
    }

    pub fn register_output(&mut self, node: IrNodeId) {
        self.outputs.insert(node);
    }

    pub fn unregister_output(&mut self, node: IrNodeId) {
        self.outputs.remove(&node);
    }

    pub fn operations<'a>(&'a self) -> Values<'a, IrOperationId, IrOperation> {
        self.ops.values()
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

    pub fn copy(&mut self, node: IrNodeId) -> Result<IrNodeId, IrError> {
        self.add_op([node], IrCopy(self.get_node_type(node)?)).map(|x| x[0])
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
            .ok_or("Cycle found!".into())
            .map(|x| x.into_iter().map(IrOperationId::from_inner).collect())
    }

    pub fn is_copy(&self, node: IrNodeId) -> Result<Option<IrNodeId>, IrError> {
        let op = self.get_op(self.get_parent_op(node)?)?;
        Ok(IrOperation::downcast::<IrCopy>(op.op()).is_some().then(|| op.inputs()[0]))
    }

    pub fn check_valid(&self) -> Result<(), IrError> {
        let mut registered_outputs = HashSet::new();
        let mut expected_child_count = HashMap::new();
        let mut actual_child_count: HashMap<_, _> = self.nodes.keys().map(|x| (x, 0)).collect();

        fn check<T: Into<String>>(cond: bool, msg: T) -> Result<(), IrError> {
            cond.then_some(()).ok_or(format!("{}!", msg.into()).into())
        }

        for op_id in self.topo_order_ops()? {
            let op = self.get_op(op_id)?;

            IrOperation::check(
                op.inputs().iter().map(|&x| self.get_node(x)).collect::<Result<Vec<_>, _>>()?,
                op.outputs().iter().map(|&x| self.get_node(x)).collect::<Result<Vec<_>, _>>()?,
                op.op().as_ref(),
            )?;

            for input in op.inputs() {
                *actual_child_count.get_mut(input).ok_or("Unexpected input node!")? += 1;
            }

            let output_types = op.op().outputs();

            check(
                op.outputs().len() == output_types.len(),
                format!(
                    "Length of operation outputs ({}) does not match expected ({})",
                    op.outputs().len(),
                    output_types.len()
                ),
            )?;

            for (&output, ty) in op.outputs().iter().zip(output_types) {
                check(registered_outputs.insert(output), "Output already registered")?;

                let node = self.get_node(output)?;
                check(node.ty() == ty, format!("Output type ({:?}) does not match expected ({ty:?})", node.ty()))?;
                check(
                    node.id() == output,
                    format!("Output id ({:?}) does not match expected ({output:?})", node.id()),
                )?;
                check(
                    expected_child_count.insert(output, node.children()).is_none(),
                    "Expected child count already present",
                )?;
            }
        }

        for (id, count) in expected_child_count {
            let actual = *actual_child_count.get(&id).ok_or("Node does not exist!")?;
            check(count == actual, format!("Actual child count ({actual}) does not match expected ({count})"))?;
        }

        Ok(())
    }

    pub fn add_op(
        &mut self,
        inputs: impl AsRef<[IrNodeId]>,
        op: impl IrOperationType,
    ) -> Result<Vec<IrNodeId>, IrError> {
        self.add_op_dyn(inputs, Rc::new(op))
    }

    pub(crate) fn add_op_dyn(
        &mut self,
        inputs: impl AsRef<[IrNodeId]>,
        op: Rc<dyn IrOperationType>,
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
            return Err("Invalid operation outputs!".into());
        }

        Ok(output_ids)
    }

    pub fn remove_op(&mut self, id: IrOperationId) -> Result<(), IrError> {
        fn check(cond: bool, msg: impl Into<String>) -> Result<(), IrError> {
            cond.then_some(()).ok_or(format!("{}!", msg.into()).into())
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

    /// Swaps the operation that each node is output from.
    /// Potentially leaves graph in a cyclic state.
    pub fn swap_outputs_unchecked(&mut self, id1: IrNodeId, id2: IrNodeId) -> Result<(), IrError> {
        let node1 = self.get_node(id1)?;
        let node2 = self.get_node(id2)?;

        if node1.ty() != node2.ty() {
            return Err(format!("Type mismatch {:?} != {:?}!", node1.ty(), node2.ty()).into());
        }

        let op1 = self.get_parent_op(id1)?;
        let op2 = self.get_parent_op(id2)?;

        if op1 == op2 {
            self.get_op_mut(op1)?.swap_output_with(id2, id1)?;
            return Ok(());
        }

        *self.links.get_mut(&id1).ok_or("Node does not exist!")? = op2;
        *self.links.get_mut(&id2).ok_or("Node does not exist!")? = op1;

        self.get_op_mut(op1)?.swap_output_with(id2, id1)?;
        self.get_op_mut(op2)?.swap_output_with(id1, id2)?;

        Ok(())
    }

    /// Replaces all instances of `old` in operation inputs by `new`.
    /// Potentially leaves graph in a cyclic state.
    pub fn replace_input_unchecked(&mut self, new: IrNodeId, old: IrNodeId) -> Result<(), IrError> {
        if self.get_node_type(new)? != self.get_node_type(old)? {
            return Err("Mismatched types!".into());
        }

        for op_id in self.ops.keys().cloned().collect::<Vec<_>>() {
            let count = self.get_op_mut(op_id)?.swap_input_with(new, old);
            self.get_node_mut(new)?.children += count;
            self.get_node_mut(old)?.children = 0;
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
                let tensor = DTypeTensor::new(ty.dtype(), size);
                let is_prev = values.contains_key(&output);

                if IrOperation::downcast::<Leaf>(op.op()).is_none() {
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
                .ok_or("Input missing!")?;

            let mut op_outputs = op
                .outputs()
                .iter()
                .map(|i| values.get(i).map(|i| i.borrow_mut()))
                .collect::<Option<Vec<_>>>()
                .ok_or("Output missing!")?;

            op.op().evaluate(
                &op_inputs.iter().map(|x| &**x).collect::<Vec<_>>(),
                &mut op_outputs.iter_mut().map(|x| &mut **x).collect::<Vec<_>>(),
            );
        }

        Ok(values.into_iter().filter_map(|x| self.is_output(x.0).then(|| (x.0, x.1.into_inner()))).collect())
    }

    pub fn as_highlighted(&self) -> String {
        let kwd = &Ansi::rgb(183, 120, 221);
        let cnt = &Ansi::rgb(86, 182, 194);
        let inp = &Ansi::rgb(209, 154, 102);
        let brk = &Ansi::rgb(37, 113, 242);
        let typ = &Ansi::rgb(229, 187, 107);

        fn rgb(s: impl fmt::Display, colour: impl fmt::Display) -> String {
            format!("{colour}{s}{}", Ansi::rgb(171, 178, 191))
        }

        let mut s = self.to_string();
        s = s.replace("[", &rgb("[", brk));
        s = s.replace("]", &rgb("]", brk));
        s = s.replace("(", &rgb("(", brk));
        s = s.replace(")", &rgb(")", brk));
        s = s.replace("{", &rgb("{", brk));
        s = s.replace("}", &rgb("}", brk));
        s = format!("{}{s}", Ansi::rgb(171, 178, 191));
        s = s.replace("irgraph", &rgb("irgraph", kwd));
        s = s.replace("return", &rgb("return", kwd));
        s = s.replace("constant", &rgb("constant", kwd));
        s = s.replace(":", &rgb(":", cnt));
        s = s.replace(".", &rgb(".", cnt));
        s = s.replace("=", &rgb("=", cnt));
        s = s.replace("f32", &rgb("f32", typ));
        s = s.replace("i32", &rgb("i32", typ));

        let mut vars = HashSet::new();
        let mut var = 0;
        let mut in_var = false;

        let _ = s.replace(
            |c: char| {
                if in_var {
                    if let Some(digit) = c.to_digit(10) {
                        var = 10 * var + digit;
                    } else {
                        vars.insert(var);
                        var = 0;
                        in_var = false;
                    }
                }

                if c == '%' {
                    in_var = true;
                }

                false
            },
            "",
        );

        let mut vars = vars.into_iter().collect::<Vec<_>>();
        vars.sort();
        vars.reverse();

        for var in vars {
            let id = IrNodeId::new(var as usize);
            let name = if self.is_input(id) {
                format!("%{}{}", Ansi::Clear, rgb(var, inp))
            } else {
                format!("%{}{var}", Ansi::rgb(171, 178, 191))
            };
            s = s.replace(&format!("{id:?}"), &name);
        }

        s = s.replace("%", &rgb("%", cnt));

        format!("{s}{}", Ansi::Clear)
    }

    pub fn as_graphviz(&self) -> Result<String, std::fmt::Error> {
        use std::fmt::Write;

        let mut s = String::new();

        let op_ids = self.topo_order_ops().unwrap();

        let get_label = |id, op| {
            let outputs = self.get_op(op).unwrap().outputs();

            if outputs.len() > 1 {
                let idx = outputs.iter().position(|&x| x == id).unwrap();
                format!(" [label=.{idx}]")
            } else {
                "".to_string()
            }
        };

        writeln!(&mut s, "digraph G {{ node [style=filled,color=lightgrey];")?;

        for op_id in op_ids {
            let op = self.get_op(op_id).unwrap();

            let opname = op.op().opname();
            let inputs = op.inputs();
            let lbl = op_id.inner();

            if inputs.is_empty() {
                writeln!(&mut s, "op{lbl} [label=\"{opname}\", style=filled, color=lightblue];")?;
            } else {
                writeln!(&mut s, "op{lbl} [label=\"{opname}\"];")?;

                for &input in inputs {
                    let parent_op = self.get_parent_op(input).unwrap();
                    let label = get_label(input, parent_op);
                    writeln!(&mut s, "op{} -> op{lbl:?}{label};", parent_op.inner())?;
                }
            }
        }

        writeln!(&mut s, "return [label=\"return\", style=filled, color=green];")?;

        for &output in &self.outputs {
            let parent_op = self.get_parent_op(output).unwrap();
            let label = get_label(output, parent_op);
            writeln!(&mut s, "op{} -> return{label};", parent_op.inner())?;
        }

        write!(&mut s, "}}")?;

        Ok(s)
    }
}

impl fmt::Display for IrGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn map<T>(x: Result<T, IrError>) -> Result<T, fmt::Error> {
            x.map_err(|_| fmt::Error)
        }

        write!(f, "irgraph(")?;
        let leaves = self.ops.values().filter(|x| IrOperation::downcast::<Leaf>(x.op()).is_some()).collect::<Vec<_>>();
        let mline = leaves.len() >= 5;

        for (i, leaf) in leaves.iter().enumerate() {
            if mline {
                writeln!(f)?;
                write!(f, "    ")?;
            } else if i != 0 {
                write!(f, ", ")?;
            }

            let node = leaf.outputs()[0];
            let ty = map(self.get_node_type(node))?;

            write!(f, "{node:?}: {ty:?}")?;
        }

        if mline {
            writeln!(f)?;
        }

        writeln!(f, ") {{")?;

        for id in map(self.topo_order_ops())? {
            let op = map(self.get_op(id))?;

            if IrOperation::downcast::<Leaf>(op.op()).is_some() {
                continue;
            }

            let inputs = op.inputs();
            let outputs = op.outputs();

            write!(f, "    ")?;
            if outputs.len() > 1 {
                write!(f, "[")?;
            }

            let output_tys =
                map(outputs.iter().map(|x| self.get_node(*x).map(IrNode::ty)).collect::<Result<Vec<_>, _>>())?;

            for (i, (&output, ty)) in outputs.iter().zip(output_tys).enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }

                write!(f, "{output:?}: {ty:?}")?;
            }

            if outputs.len() > 1 {
                write!(f, "]")?;
            }

            write!(f, " = {}(", op.op().opname())?;

            for (i, &input) in inputs.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }

                write!(f, "{input:?}")?;
            }

            writeln!(f, ")")?;
        }

        write!(f, "    return ")?;
        for (i, &output) in self.outputs.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }

            write!(f, "{output:?}")?;
        }

        writeln!(f)?;
        write!(f, "}}")?;

        Ok(())
    }
}
