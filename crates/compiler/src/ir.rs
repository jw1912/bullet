//! The core SSA IR that is reused across the compiler

mod node;
mod operation;
mod topo;

use std::{
    collections::{HashMap, HashSet, hash_map::Values},
    fmt,
};

pub use node::{Node, NodeId};
pub use operation::{Op, OpId};

/// Required for type checking the IR and graphviz output
pub trait Operation<T>: fmt::Debug {
    fn opname(&self) -> String;
    fn inputs(&self) -> Vec<T>;
    fn outputs(&self) -> Vec<T>;
}

/// Defines the type system for nodes in the IR
pub trait TypeSystem: Copy + fmt::Debug {
    type Type: Clone + fmt::Debug + Eq;
    type OpData: Clone + Operation<Self::Type> + fmt::Debug;
}

/// Simple string error type tied to the IR
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IRError(String);

impl<T: Into<String>> From<T> for IRError {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

/// The core SSA IR that is reused across the compiler,
/// generic over the type system
#[derive(Clone, Default, Debug)]
pub struct IR<T: TypeSystem> {
    nodes: HashMap<NodeId, Node<T>>,
    ops: HashMap<OpId, Op<T>>,
    links: HashMap<NodeId, OpId>,
}

impl<T: TypeSystem> IR<T> {
    /// Reference to the node with given ID
    pub fn node(&self, node: NodeId) -> Result<&Node<T>, IRError> {
        self.nodes.get(&node).ok_or(format!("Node<T> {node:?} does not exist!").into())
    }

    /// Mutable reference to the node with given ID
    pub fn node_mut(&mut self, node: NodeId) -> Result<&mut Node<T>, IRError> {
        self.nodes.get_mut(&node).ok_or(format!("Node<T> {node:?} does not exist!").into())
    }

    /// Unordered iterator over the nodes in the graph
    pub fn nodes<'a>(&'a self) -> Values<'a, NodeId, Node<T>> {
        self.nodes.values()
    }

    /// Reference to the operation with given ID
    pub fn op(&self, op: OpId) -> Result<&Op<T>, IRError> {
        self.ops.get(&op).ok_or(format!("Operation {op:?} does not exist!").into())
    }

    /// Mutable reference to the operation with given ID
    pub fn op_mut(&mut self, op: OpId) -> Result<&mut Op<T>, IRError> {
        self.ops.get_mut(&op).ok_or(format!("Operation {op:?} does not exist!").into())
    }

    /// Unordered iterator over the operations in the graph
    pub fn operations<'a>(&'a self) -> Values<'a, OpId, Op<T>> {
        self.ops.values()
    }

    /// Get the the parent operation of this node
    pub fn parent_op(&self, node: NodeId) -> Result<OpId, IRError> {
        self.links.get(&node).cloned().ok_or(format!("Node<T> {node:?} does not exist!").into())
    }

    /// Returns the graph operation IDs in topological order
    pub fn topo_order_ops(&self) -> Result<Vec<OpId>, IRError> {
        let edges_rev = self
            .operations()
            .map(|data| {
                data.inputs()
                    .iter()
                    .map(|&x| self.parent_op(x).map(|x| x.inner()))
                    .collect::<Result<_, _>>()
                    .map(|x| (data.id().inner(), x))
            })
            .collect::<Result<_, _>>()?;

        topo::topo_order(edges_rev).ok_or("Cycle found!".into()).map(|x| x.into_iter().map(OpId::from_inner).collect())
    }

    /// Returns an error if any graph invariants are broken, otherwise returns `Ok(())`
    pub fn check_valid(&self) -> Result<(), IRError> {
        let mut registered_outputs = HashSet::new();
        let mut expected_child_count = HashMap::new();
        let mut actual_child_count: HashMap<_, _> = self.nodes().map(|x| (x.id(), 0)).collect();

        fn check<T: Into<String>>(cond: bool, msg: T) -> Result<(), IRError> {
            cond.then_some(()).ok_or(format!("{}!", msg.into()).into())
        }

        for op_id in self.topo_order_ops()? {
            let op = self.op(op_id)?;

            Op::check(
                op.inputs().iter().map(|&x| self.node(x)).collect::<Result<Vec<_>, _>>()?,
                op.outputs().iter().map(|&x| self.node(x)).collect::<Result<Vec<_>, _>>()?,
                op.data(),
            )?;

            for input in op.inputs() {
                *actual_child_count.get_mut(input).ok_or("Unexpected input node!")? += 1;
            }

            let output_types = op.data().outputs();

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

                let node = self.node(output)?;
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

    /// Adds a new operation to the graph
    pub fn add_op(&mut self, inputs: impl AsRef<[NodeId]>, data: T::OpData) -> Result<Vec<NodeId>, IRError> {
        let output_ids = (0..data.outputs().len()).map(|_| NodeId::default()).collect::<Vec<_>>();
        let output_tys = data.outputs();

        let mut error = false;

        for (&out_id, out_ty) in output_ids.iter().zip(output_tys.iter()) {
            error |= self.nodes.insert(out_id, Node::new(out_id, out_ty.clone())).is_some();
        }

        let inputs = inputs.as_ref().iter().map(|&id| self.node(id)).collect::<Result<_, _>>()?;
        let outputs = output_ids.iter().map(|&id| self.node(id)).collect::<Result<_, _>>()?;
        let op = Op::new(inputs, outputs, data)?;
        let op_id = op.id();

        for &input in op.inputs() {
            self.node_mut(input)?.inc_children();
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

    /// Removes operation - fails if any of the output nodes of this operation have children
    pub fn remove_op(&mut self, id: OpId) -> Result<(), IRError> {
        fn check(cond: bool, msg: impl Into<String>) -> Result<(), IRError> {
            cond.then_some(()).ok_or(format!("{}!", msg.into()).into())
        }

        for output in self.op(id)?.outputs().to_vec() {
            let node = self.nodes.remove(&output).expect("Node<T> must be present `nodes`!");
            check(node.children() == 0, format!("node {node:?} has children"))?;

            let op_id = self.links.remove(&output).expect("Node<T> must be present `links`!");
            check(op_id == id, format!("operation id mismatch ({op_id:?} != {id:?})"))?;
        }

        for input in self.op(id)?.inputs().to_vec() {
            self.node_mut(input)?.dec_children();
        }

        self.ops.remove(&id).expect("Already verified node is present `ops`!");

        Ok(())
    }

    /// Swaps the operation that each node is output from
    pub fn swap_outputs_no_cycle_check(&mut self, id1: NodeId, id2: NodeId) -> Result<(), IRError> {
        let node1 = self.node(id1)?;
        let node2 = self.node(id2)?;

        if node1.ty() != node2.ty() {
            return Err(format!("Type mismatch {:?} != {:?}!", node1.ty(), node2.ty()).into());
        }

        let op1 = self.parent_op(id1)?;
        let op2 = self.parent_op(id2)?;

        if op1 == op2 {
            self.op_mut(op1)?.swap_output_with(id2, id1)?;
            return Ok(());
        }

        *self.links.get_mut(&id1).ok_or("Node<T> does not exist!")? = op2;
        *self.links.get_mut(&id2).ok_or("Node<T> does not exist!")? = op1;

        self.op_mut(op1)?.swap_output_with(id2, id1)?;
        self.op_mut(op2)?.swap_output_with(id1, id2)?;

        Ok(())
    }

    /// Replaces all instances of `old` in operation inputs by `new`
    pub fn replace_input_no_cycle_check(&mut self, new: NodeId, old: NodeId) -> Result<(), IRError> {
        if self.node(new)?.ty() != self.node(old)?.ty() {
            return Err("Mismatched types!".into());
        }

        for op_id in self.ops.keys().cloned().collect::<Vec<_>>() {
            let count = self.op_mut(op_id)?.swap_input_with(new, old);
            self.node_mut(new)?.children += count;
            self.node_mut(old)?.children = 0;
        }

        Ok(())
    }

    /// Finds all operations that are required to complete before this one can be performed
    pub fn get_dependent_ops_set(&self, id: OpId) -> Result<HashSet<OpId>, IRError> {
        let mut set: HashSet<_> = [id].into();

        for parent in self.op(id)?.inputs() {
            let parent_op = self.parent_op(*parent)?;

            for op in self.get_dependent_ops_set(parent_op)? {
                set.insert(op);
            }
        }

        Ok(set)
    }
}

impl<T: TypeSystem> fmt::Display for IR<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op_ids = self.topo_order_ops().unwrap();

        let get_label = |id, op| {
            let outputs = self.op(op).unwrap().outputs();

            if outputs.len() > 1 {
                let idx = outputs.iter().position(|&x| x == id).unwrap();
                format!(" [label=.{idx}]")
            } else {
                "".to_string()
            }
        };

        writeln!(f, "digraph G {{ node [style=filled,color=lightgrey];")?;

        for op_id in op_ids {
            let op = self.op(op_id).unwrap();

            let opname = op.data().opname();
            let inputs = op.inputs();
            let lbl = op_id.inner();

            if inputs.is_empty() {
                writeln!(f, "op{lbl} [label=\"{opname}\", style=filled, color=lightblue];")?;
            } else {
                writeln!(f, "op{lbl} [label=\"{opname}\"];")?;

                for &input in inputs {
                    let parent_op = self.parent_op(input).unwrap();
                    let label = get_label(input, parent_op);
                    writeln!(f, "op{} -> op{lbl:?}{label};", parent_op.inner())?;
                }
            }
        }

        write!(f, "}}")
    }
}
