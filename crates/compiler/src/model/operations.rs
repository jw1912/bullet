use crate::{
    ir::{IRError, NodeId},
    model::{Layout, MType, ModelOperation},
    tensor::{DType, TType, TensorIR},
};

#[derive(Clone, Copy, Debug)]
pub struct Input(MType);
impl ModelOperation for Input {
    fn opname(&self) -> String {
        let MType { batch, rows, cols, layout } = self.0;
        format!("Input<{layout:?}, {}{rows}x{cols}>", if batch { "Bx" } else { "" })
    }

    fn inputs(&self) -> Vec<MType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn lower(&self, batch_size: usize, tensor: &mut TensorIR, _inputs: Vec<NodeId>) -> Result<Vec<NodeId>, IRError> {
        let MType { batch, rows, cols, layout } = self.0;

        let (mut size, dtype) = match layout {
            Layout::Dense(dtype) => (rows * cols, dtype),
            Layout::Sparse(nnz) => (nnz * cols, DType::I32),
        };

        if batch {
            size *= batch_size;
        }

        Ok(vec![tensor.add_input(TType::new(size, dtype))])
    }

    fn gradient(&self, _output_grads: Vec<NodeId>) -> Result<Vec<Option<NodeId>>, IRError> {
        Ok(Vec::new())
    }
}
