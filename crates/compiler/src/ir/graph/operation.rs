use std::{
    collections::HashSet,
    fmt::Debug,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    core::DTypeTensor,
    ir::graph::{IrError, IrNode, IrNodeId, IrType},
};

pub trait IrOperationType: std::any::Any + Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<IrType>;

    fn outputs(&self) -> Vec<IrType>;

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]);

    fn equals(&self, other: &Rc<dyn IrOperationType>) -> bool;

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        Vec::new()
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct IrOperationId(usize);

impl Default for IrOperationId {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl IrOperationId {
    pub(super) fn from_inner(id: usize) -> Self {
        Self(id)
    }

    pub fn inner(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct IrOperation {
    id: IrOperationId,
    inputs: Vec<IrNodeId>,
    outputs: Vec<IrNodeId>,
    op: Rc<dyn IrOperationType>,
}

impl IrOperation {
    pub fn new(inputs: Vec<&IrNode>, outputs: Vec<&IrNode>, op: Rc<dyn IrOperationType>) -> Result<Self, IrError> {
        Self::check(&inputs, &outputs, op.as_ref())?;
        let id = IrOperationId::default();
        let inputs = inputs.iter().map(|&i| i.id()).collect();
        let outputs = outputs.iter().map(|&i| i.id()).collect();

        Ok(Self { id, op, inputs, outputs })
    }

    pub fn check<'a>(
        inputs: impl AsRef<[&'a IrNode]>,
        outputs: impl AsRef<[&'a IrNode]>,
        op: &'a dyn IrOperationType,
    ) -> Result<(), IrError> {
        if op.inputs() != inputs.as_ref().iter().map(|&i| i.ty()).collect::<Vec<_>>() {
            return Err("IrOperation::new: inputs don't match expected!".into());
        }

        if op.outputs() != outputs.as_ref().iter().map(|&i| i.ty()).collect::<Vec<_>>() {
            return Err("IrOperation::new: outputs don't match expected!".into());
        }

        Ok(())
    }

    pub fn id(&self) -> IrOperationId {
        self.id
    }

    pub fn inputs(&self) -> &[IrNodeId] {
        &self.inputs
    }

    pub fn set_input(&mut self, idx: usize, node: IrNodeId) {
        self.inputs[idx] = node;
    }

    pub fn outputs(&self) -> &[IrNodeId] {
        &self.outputs
    }

    pub fn swap_input_with(&mut self, new: IrNodeId, old: IrNodeId) -> usize {
        let mut count = 0;

        for id in &mut self.inputs {
            if *id == old {
                *id = new;
                count += 1;
            }
        }

        count
    }

    pub fn swap_output_with(&mut self, new: IrNodeId, old: IrNodeId) -> Result<(), IrError> {
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

        found.then_some(()).ok_or(format!("IrOperation::swap_output_with: {old:?} not found!").into())
    }

    pub fn op(&self) -> &Rc<dyn IrOperationType> {
        &self.op
    }

    pub fn downcast_rc<T: IrOperationType + 'static>(input: &Rc<dyn IrOperationType>) -> Option<&T> {
        let op: &dyn std::any::Any = input.as_ref();
        op.downcast_ref::<T>()
    }

    pub fn downcast<T: IrOperationType>(&self) -> Option<&T> {
        Self::downcast_rc::<T>(&self.op)
    }

    /// Canonicalise ordering of commutative inputs.
    pub fn order_commutative_inputs(&mut self) -> Result<(), IrError> {
        let groups = self.op.commutating_groups();

        for (i, group_i) in groups.iter().enumerate() {
            for group_j in groups.iter().skip(i + 1) {
                if group_i.intersection(group_j).next().is_some() {
                    return Err("Distinct commutating groups intersect!".into());
                }
            }
        }

        for group in groups {
            let mut group = group.into_iter().collect::<Vec<_>>();
            let mut nodes = group.iter().map(|&i| self.inputs[i]).collect::<Vec<_>>();

            if self.op.inputs().iter().collect::<HashSet<_>>().len() > 1 {
                return Err("Inputs within commutating group have differing types!".into());
            }

            group.sort();
            nodes.sort();

            for (idx, id) in group.into_iter().zip(nodes) {
                self.set_input(idx, id);
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct IrInput(pub IrType);

impl IrOperationType for IrInput {
    fn opname(&self) -> String {
        format!("leaf<{:?}>", self.0)
    }

    fn inputs(&self) -> Vec<IrType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![self.0]
    }

    fn evaluate(&self, _: &[&DTypeTensor], _: &mut [&mut DTypeTensor]) {}

    fn equals(&self, _: &Rc<dyn IrOperationType>) -> bool {
        false
    }
}
