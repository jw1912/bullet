use std::sync::atomic::{AtomicUsize, Ordering};

use super::{IRError, Node, NodeId, Operation, TypeSystem};

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

    pub(super) fn inner(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct Op<T: TypeSystem> {
    id: OpId,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
    data: T::OpData,
}

impl<T: TypeSystem> Op<T> {
    pub fn new(inputs: Vec<&Node<T>>, outputs: Vec<&Node<T>>, data: T::OpData) -> Result<Self, IRError> {
        Self::check(&inputs, &outputs, &data)?;
        let id = OpId::default();
        let inputs = inputs.iter().map(|&i| i.id()).collect();
        let outputs = outputs.iter().map(|&i| i.id()).collect();

        Ok(Self { id, data, inputs, outputs })
    }

    pub fn check<'a>(
        inputs: impl AsRef<[&'a Node<T>]>,
        outputs: impl AsRef<[&'a Node<T>]>,
        data: &'a T::OpData,
    ) -> Result<(), IRError>
    where
        T: 'a,
    {
        if data.inputs() != inputs.as_ref().iter().map(|&i| i.ty()).collect::<Vec<_>>() {
            return Err("Operation inputs don't match expected!".into());
        }

        if data.outputs() != outputs.as_ref().iter().map(|&i| i.ty()).collect::<Vec<_>>() {
            return Err("Operation outputs don't match expected!".into());
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

    pub fn swap_output_with(&mut self, new: NodeId, old: NodeId) -> Result<(), IRError> {
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

    pub fn data(&self) -> &T::OpData {
        &self.data
    }
}
