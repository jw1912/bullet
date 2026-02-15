use std::{collections::HashSet, rc::Rc};

use bullet_compiler::graph::{DType, OpType, Size, TType, TValue};

#[derive(Clone, Copy, Debug)]
pub(super) struct VarSizeStub(pub Size);

impl OpType for VarSizeStub {
    fn opname(&self) -> String {
        "gpu.stub.varsize".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<TType> {
        vec![TType::new(self.0, DType::I32)]
    }

    fn equals(&self, _other: &Rc<dyn OpType>) -> bool {
        false
    }

    fn evaluate(&self, _inputs: Vec<&TValue>, _outputs: Vec<&mut TValue>) {
        unimplemented!()
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        Vec::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ThreadIdxStub(pub Size);

impl OpType for ThreadIdxStub {
    fn opname(&self) -> String {
        "gpu.stub.tid".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.0, DType::I32)]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![TType::new(self.0, DType::I32)]
    }

    fn equals(&self, _other: &Rc<dyn OpType>) -> bool {
        false
    }

    fn evaluate(&self, _inputs: Vec<&TValue>, _outputs: Vec<&mut TValue>) {
        unimplemented!()
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        Vec::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReadStub(pub(super) TType, pub(super) Size);

impl OpType for ReadStub {
    fn opname(&self) -> String {
        "gpu.stub.read".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.0, TType::new(self.1, DType::I32)]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![TType::new(self.1, self.0.dtype())]
    }

    fn equals(&self, _other: &Rc<dyn OpType>) -> bool {
        false
    }

    fn evaluate(&self, _inputs: Vec<&TValue>, _outputs: Vec<&mut TValue>) {
        unimplemented!()
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        Vec::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WriteStub(pub(super) TType);

impl OpType for WriteStub {
    fn opname(&self) -> String {
        "gpu.stub.write".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.0, self.0, TType::new(self.0.size(), DType::I32)]
    }

    fn outputs(&self) -> Vec<TType> {
        Vec::new()
    }

    fn equals(&self, _other: &Rc<dyn OpType>) -> bool {
        false
    }

    fn evaluate(&self, _inputs: Vec<&TValue>, _outputs: Vec<&mut TValue>) {
        unimplemented!()
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        Vec::new()
    }
}

#[derive(Clone, Debug)]
pub struct ComputeStub {
    inputs: Vec<TType>,
    outputs: Vec<TType>,
    code: String,
}

impl ComputeStub {
    pub fn new(inputs: impl Into<Vec<TType>>, outputs: impl Into<Vec<TType>>, code: impl Into<String>) -> Self {
        Self { inputs: inputs.into(), outputs: outputs.into(), code: code.into() }
    }
}

impl OpType for ComputeStub {
    fn opname(&self) -> String {
        "gpu.stub.compute".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        self.inputs.clone()
    }

    fn outputs(&self) -> Vec<TType> {
        self.outputs.clone()
    }

    fn equals(&self, _other: &Rc<dyn OpType>) -> bool {
        false
    }

    fn evaluate(&self, _inputs: Vec<&TValue>, _outputs: Vec<&mut TValue>) {
        unimplemented!()
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        Vec::new()
    }
}
