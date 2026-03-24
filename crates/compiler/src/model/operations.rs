use crate::{
    ir::{IRError, NodeId},
    model::{Layout, MType, ModelIR, ModelOperation},
    tensor::{DType, TType, TensorIR},
};

#[derive(Clone, Copy, Debug)]
pub struct Input(pub(super) MType);
impl ModelOperation for Input {
    fn opname(&self) -> String {
        let MType { batch, rows, cols, layout } = self.0;
        format!("Input<{layout:?}, {}{rows}x{cols}>", if batch { "Bx" } else { "" })
    }

    fn inputs(&self) -> Vec<MType> {
        Vec::new()
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, _inputs: Vec<NodeId>) -> Result<Vec<NodeId>, IRError> {
        let MType { batch, rows, cols, layout } = self.0;

        let (mut size, dtype) = match layout {
            Layout::Dense(dtype) => (rows * cols, dtype),
            Layout::Sparse(nnz) => (nnz * cols, DType::I32),
        };

        if batch {
            size *= batch_size;
        }

        Ok(vec![lower.add_input(TType::new(size, dtype))])
    }

    fn gradient(&self, _ir: &mut ModelIR, _output_grad: NodeId) -> Result<Vec<Option<NodeId>>, IRError> {
        Ok(Vec::new())
    }
}
