use crate::{
    ir::NodeId,
    model::{Layout, MType, ModelOperation, ModelIR},
    tensor::{IRTrace, TValue, TensorIR},
};

#[derive(Clone, Copy, Debug)]
pub struct Input(pub(crate) MType);
impl ModelOperation for Input {
    fn opname(&self) -> String {
        format!("Input<{}>", self.0)
    }

    fn inputs(&self) -> Vec<MType> {
        Vec::new()
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, _inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        Ok(lower.add_input(self.0.ttype(batch_size)))
    }

    fn gradient(
        &self,
        _ir: &mut ModelIR,
        _inputs: Vec<NodeId>,
        _output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        Ok(Vec::new())
    }
}

#[derive(Debug)]
pub struct Constant(MType, TValue);

impl Constant {
    pub fn new(value: TValue, rows: usize, cols: usize) -> Self {
        assert_eq!(value.size(), rows * cols);
        Self(MType { batch: false, rows, cols, layout: Layout::Dense(value.dtype()) }, value)
    }
}

impl ModelOperation for Constant {
    fn opname(&self) -> String {
        format!("Constant<{}>", self.0)
    }

    fn inputs(&self) -> Vec<MType> {
        Vec::new()
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, _batch_size: usize, lower: &mut TensorIR, _inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        Ok(lower.add_const(self.1.clone()))
    }

    fn gradient(
        &self,
        _ir: &mut ModelIR,
        _inputs: Vec<NodeId>,
        _output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        Ok(Vec::new())
    }
}
