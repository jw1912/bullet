use crate::{
    ir::NodeId,
    model::{Layout, MType, ModelOperation},
    tensor::{
        DType, DValue, IRTrace, TensorIR,
        operation::{CABinary, PadAcrossDimension, Select, SliceAcrossDimension},
    },
};

#[derive(Clone, Copy, Debug)]
pub struct Slice(MType, usize, usize, bool);

impl Slice {
    pub fn new(ty: MType, start: usize, end: usize, across_rows: bool) -> Self {
        assert!(end > start);
        assert!(end <= if across_rows { ty.rows } else { ty.cols });
        assert!(ty.is_dense());

        Self(ty, start, end, across_rows)
    }
}

impl ModelOperation for Slice {
    fn opname(&self) -> String {
        let Slice(_, start, end, rows) = *self;
        format!("Pad<{start}, {end}, {rows}>")
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
}

#[derive(Clone, Copy, Debug)]
pub struct Concat(DType, usize, usize, bool);

impl Concat {
    pub fn new(dtype: DType, rows_lhs: usize, rows_rhs: usize, batched: bool) -> Self {
        Self(dtype, rows_lhs, rows_rhs, batched)
    }
}

impl ModelOperation for Concat {
    fn opname(&self) -> String {
        "Concat".to_string()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![
            MType { rows: self.1, cols: 1, batch: self.3, layout: Layout::Dense(self.0) },
            MType { rows: self.2, cols: 1, batch: self.3, layout: Layout::Dense(self.0) },
        ]
    }

    fn output(&self) -> MType {
        MType { rows: self.1 + self.2, cols: 1, batch: self.3, layout: Layout::Dense(self.0) }
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let batch = if self.3 { batch_size } else { 1 };
        let pad1 = PadAcrossDimension::new([batch, self.1], 1, 0, self.2, DValue::zero(self.0));
        let pad2 = PadAcrossDimension::new([batch, self.2], 1, self.1, 0, DValue::zero(self.0));
        let p1 = lower.add_op(&inputs[..1], pad1)?[0];
        let p2 = lower.add_op(&inputs[1..], pad2)?[0];
        lower.add_binary(p1, p2, CABinary::Add)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SelectRows(DType, usize, usize, bool);

impl SelectRows {
    pub fn new(dtype: DType, rows: usize, divisor: usize, batch: bool) -> Self {
        assert!(rows.is_multiple_of(divisor));
        Self(dtype, rows, divisor, batch)
    }
}

impl ModelOperation for SelectRows {
    fn opname(&self) -> String {
        "Select".to_string()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![
            MType { rows: self.1, cols: 1, batch: self.3, layout: Layout::Dense(self.0) },
            MType { rows: self.2, cols: 1, batch: self.3, layout: Layout::Sparse(1) },
        ]
    }

    fn output(&self) -> MType {
        MType { rows: self.1 / self.2, cols: 1, batch: self.3, layout: Layout::Dense(self.0) }
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let batch = if self.3 { batch_size } else { 1 }.into();
        let select = Select { dtype: self.0, batch, inner: self.1.into(), divisor: self.2.into() };
        lower.add_op(inputs, Ok::<_, IRTrace>(select)).map(|x| x[0])
    }
}
