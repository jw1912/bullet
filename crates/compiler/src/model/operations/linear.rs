use crate::{
    ir::NodeId,
    model::{Layout, MType, ModelOperation, ModelIR},
    tensor::{
        DType, IRTrace, TensorIR,
        operation::{self, MatrixLayout, Reduction},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    Batch,
    Rows,
    Cols,
}

#[derive(Clone, Copy, Debug)]
pub struct Broadcast(pub MType, pub Dim, pub Option<usize>);
impl ModelOperation for Broadcast {
    fn opname(&self) -> String {
        let reps = self.2.map(|r| format!("x{r}")).unwrap_or_default();
        format!("Broadcast<{:?}{reps}>", self.1)
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        match self.1 {
            Dim::Batch => MType { batch: true, ..self.0 },
            Dim::Cols => MType { cols: self.0.cols * self.2.unwrap(), ..self.0 },
            Dim::Rows => MType { rows: self.0.rows * self.2.unwrap(), ..self.0 },
        }
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let input = inputs[0];
        let ty = self.0;
        let reps = self.2.unwrap_or(1);

        if self.0.batch {
            match self.1 {
                Dim::Batch => Ok(input),
                Dim::Cols => lower.add_broadcast(input, [batch_size, ty.single_size()], 1, reps),
                Dim::Rows => lower.add_broadcast(input, [batch_size * ty.cols, ty.rows], 1, reps),
            }
        } else {
            match self.1 {
                Dim::Batch => lower.add_broadcast(input, [self.0.single_size()], 0, batch_size),
                Dim::Cols => lower.add_broadcast(input, [ty.single_size()], 0, reps),
                Dim::Rows => lower.add_broadcast(input, [ty.cols, ty.rows], 1, reps),
            }
        }
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

#[derive(Clone, Copy, Debug)]
pub struct Reduce(pub MType, pub Dim, pub Reduction);
impl ModelOperation for Reduce {
    fn opname(&self) -> String {
        format!("Reduce{:?}<{:?}>", self.2, self.1)
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        match self.1 {
            Dim::Batch => MType { batch: false, ..self.0 },
            Dim::Cols => MType { cols: 1, ..self.0 },
            Dim::Rows => MType { rows: 1, ..self.0 },
        }
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let input = inputs[0];
        let ty = self.0;

        if self.0.batch {
            match self.1 {
                Dim::Batch => lower.add_reduction(input, [batch_size, ty.single_size()], 0, self.2),
                Dim::Cols => lower.add_reduction(input, [batch_size, ty.cols, ty.rows], 1, self.2),
                Dim::Rows => lower.add_reduction(input, [batch_size * ty.cols, ty.rows], 2, self.2),
            }
        } else {
            match self.1 {
                Dim::Batch => Ok(input),
                Dim::Cols => lower.add_reduction(input, [ty.cols, ty.rows], 0, self.2),
                Dim::Rows => lower.add_reduction(input, [ty.cols, ty.rows], 1, self.2),
            }
        }
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

#[derive(Clone, Copy, Debug)]
pub struct Matmul {
    pub(crate) lbatch: bool,
    pub(crate) rbatch: bool,
    pub(crate) dtype: DType,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl ModelOperation for Matmul {
    fn opname(&self) -> String {
        let Matmul { lbatch, rbatch, dtype, m, n, k } = *self;
        let lbatch = if lbatch { "B" } else { "N" };
        let rbatch = if rbatch { "B" } else { "N" };
        format!("Matmul<{dtype:?}, {lbatch}{rbatch}, {m}x{n}x{}>", k)
    }

    fn inputs(&self) -> Vec<MType> {
        let Matmul { lbatch, rbatch, dtype, m, n, k } = *self;
        vec![
            MType { batch: lbatch, rows: m, cols: n, layout: Layout::Dense(dtype) },
            MType { batch: rbatch, rows: n, cols: k, layout: Layout::Dense(dtype) },
        ]
    }

    fn output(&self) -> MType {
        let Matmul { lbatch, rbatch, dtype, m, k, .. } = *self;
        MType { batch: lbatch | rbatch, rows: m, cols: k, layout: Layout::Dense(dtype) }
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let Matmul { lbatch, rbatch, dtype, m, n, k } = *self;
        let lhs = MatrixLayout { rows: m.into(), cols: n.into(), col_mjr: true };

        let matmul = match (lbatch, rbatch) {
            (false, false) => {
                let rhs = MatrixLayout { rows: n.into(), cols: k.into(), col_mjr: true };
                operation::Matmul::new(dtype, 1, lhs, rhs)
            }
            (true, true) => {
                let rhs = MatrixLayout { rows: n.into(), cols: k.into(), col_mjr: true };
                operation::Matmul::new(dtype, batch_size, lhs, rhs)
            }
            (false, true) => {
                let rhs = MatrixLayout { rows: n.into(), cols: (k * batch_size).into(), col_mjr: true };
                operation::Matmul::new(dtype, 1, lhs, rhs)
            }
            (true, false) => unimplemented!(),
        };

        lower.add_op(inputs, matmul).map(|x| x[0])
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
