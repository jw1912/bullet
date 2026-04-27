use crate::{
    ir::NodeId,
    model::{MType, ModelOperation, ModelIR},
    tensor::{IRTrace, TensorIR, operation::SliceAcrossDimension},
};

#[derive(Clone, Copy, Debug)]
pub struct Slice(MType, usize, usize, bool);

impl Slice {
    pub fn new(ty: MType, start: usize, end: usize, across_rows: bool) -> Self {
        assert!(end > start);
        assert!(end < if across_rows { ty.rows } else { ty.cols });
        assert!(ty.is_dense());

        Self(ty, start, end, across_rows)
    }
}

impl ModelOperation for Slice {
    fn opname(&self) -> String {
        "Slice".to_string()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        if self.3 {
            MType { rows: self.2 - self.1, ..self.0 }
        } else {
            MType { cols: self.2 - self.1, ..self.0 }
        }
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let dtype = self.0.ttype(batch_size).dtype();
        let shape = [if self.0.batch { batch_size } else { 1 }, self.0.cols, self.0.rows];
        let slice = SliceAcrossDimension::new(dtype, shape, 1 + usize::from(self.3), self.1, self.2);
        lower.add_op(inputs, slice).map(|x| x[0])
    }

    fn gradient(
        &self,
        _ir: &mut ModelIR,
        _inputs: Vec<NodeId>,
        _output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        unimplemented!()
    }
}
