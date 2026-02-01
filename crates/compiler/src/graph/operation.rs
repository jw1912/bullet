use std::{
    any::Any,
    collections::HashSet,
    fmt::Debug,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::graph::{GraphError, Node, NodeId, TType, TValue};

pub trait OpType: Any + Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<TType>;

    fn outputs(&self) -> Vec<TType>;

    fn evaluate(&self, inputs: Vec<&TValue>, outputs: Vec<&mut TValue>);

    fn equals(&self, other: &Rc<dyn OpType>) -> bool;

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        Vec::new()
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct OpId(usize);

impl Default for OpId {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl OpId {
    pub(super) fn from_inner(id: usize) -> Self {
        Self(id)
    }

    pub fn inner(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct Op {
    id: OpId,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
    op: Rc<dyn OpType>,
}

impl Op {
    pub fn new(inputs: Vec<&Node>, outputs: Vec<&Node>, op: Rc<dyn OpType>) -> Result<Self, GraphError> {
        Self::check(&inputs, &outputs, op.as_ref())?;
        let id = OpId::default();
        let inputs = inputs.iter().map(|&i| i.id()).collect();
        let outputs = outputs.iter().map(|&i| i.id()).collect();

        Ok(Self { id, op, inputs, outputs })
    }

    pub fn check<'a>(
        inputs: impl AsRef<[&'a Node]>,
        outputs: impl AsRef<[&'a Node]>,
        op: &'a dyn OpType,
    ) -> Result<(), GraphError> {
        if op.inputs() != inputs.as_ref().iter().map(|&i| i.ty()).collect::<Vec<_>>() {
            return Err("Op::new: inputs don't match expected!".into());
        }

        if op.outputs() != outputs.as_ref().iter().map(|&i| i.ty()).collect::<Vec<_>>() {
            return Err("Op::new: outputs don't match expected!".into());
        }

        Ok(())
    }

    pub fn id(&self) -> OpId {
        self.id
    }

    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    pub fn set_input(&mut self, idx: usize, node: NodeId) {
        self.inputs[idx] = node;
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub fn swap_input_with(&mut self, new: NodeId, old: NodeId) -> usize {
        let mut count = 0;

        for id in &mut self.inputs {
            if *id == old {
                *id = new;
                count += 1;
            }
        }

        count
    }

    pub fn swap_output_with(&mut self, new: NodeId, old: NodeId) -> Result<(), GraphError> {
        let mut found = false;

        for id in &mut self.outputs {
            if *id == old {
                if found {
                    panic!("This cannot happen!");
                }

                *id = new;
                found = true;
            }
        }

        found.then_some(()).ok_or(format!("Op::swap_output_with: {old:?} not found!").into())
    }

    pub fn op(&self) -> &Rc<dyn OpType> {
        &self.op
    }

    pub fn downcast_rc<T: OpType>(input: &Rc<dyn OpType>) -> Option<&T> {
        let op: &dyn Any = input.as_ref();
        op.downcast_ref::<T>()
    }

    pub fn downcast<T: OpType>(&self) -> Option<&T> {
        Self::downcast_rc::<T>(&self.op)
    }

    pub fn is_input(&self) -> bool {
        self.downcast::<Input>().is_some()
    }
}

#[derive(Debug)]
pub struct Input(pub TType);

impl OpType for Input {
    fn opname(&self) -> String {
        format!("leaf<{:?}>", self.0)
    }

    fn inputs(&self) -> Vec<TType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<TType> {
        vec![self.0]
    }

    fn evaluate(&self, _: Vec<&TValue>, _: Vec<&mut TValue>) {}

    fn equals(&self, _: &Rc<dyn OpType>) -> bool {
        false
    }
}
