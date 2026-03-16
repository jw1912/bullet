use crate::{
    ir::IRError,
    tensor::{DType, DValue, OpType, Size, TType, TValue, TensorOp, operation::CABinary},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseMatmul {
    dtype: DType,
    batch: Size,
    rows: usize,
    cols: usize,
    nnz: usize,
}

impl SparseMatmul {
    pub fn new(dtype: DType, batch: impl Into<Size>, rows: usize, cols: usize, nnz: usize) -> Self {
        SparseMatmul { dtype, batch: batch.into(), rows, cols, nnz }
    }

    pub fn invert(&self) -> SparseMatmulBwd {
        SparseMatmulBwd(*self)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn batch(&self) -> Size {
        self.batch
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn nnz(&self) -> usize {
        self.nnz
    }
}

impl OpType for SparseMatmul {
    fn opname(&self) -> String {
        let SparseMatmul { batch, rows, cols, nnz, .. } = *self;
        format!("sparse.matmul<{batch:?}, {rows:?}x{cols:?}, {nnz}>")
    }

    fn inputs(&self) -> Vec<TType> {
        let SparseMatmul { dtype, batch, rows, cols, nnz } = *self;
        vec![TType::new(rows * cols, dtype), TType::new(batch * nnz, DType::I32)]
    }

    fn outputs(&self) -> Vec<TType> {
        let SparseMatmul { dtype, batch, rows, .. } = *self;
        vec![TType::new(batch * rows, dtype)]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
        let SparseMatmul { dtype, batch, rows, cols, nnz } = *self;

        let d = inputs[0];
        let s = inputs[1];
        let o = &mut outputs[0];

        let var_size = (batch * nnz).get_var_size(s.size()).unwrap_or(1);

        let b = batch.evaluate(var_size);

        assert_eq!(s.size(), b * nnz);
        assert_eq!(d.size(), rows * cols);
        assert_eq!(o.size(), b * rows);

        for bi in 0..b {
            for ri in 0..rows {
                let mut sum = DValue::zero(dtype);

                for ni in 0..nnz {
                    let DValue::I32(idx) = s.read(nnz * bi + ni) else { panic!() };
                    if idx >= 0 && (idx as usize) < cols {
                        sum = CABinary::Add.evaluate(sum, d.read(rows * idx as usize + ri)).unwrap();
                    }
                }

                o.write(rows * bi + ri, sum);
            }
        }

        true
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(other) = other.downcast::<Self>() { self == other } else { false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseMatmulBwd(pub SparseMatmul);

impl OpType for SparseMatmulBwd {
    fn opname(&self) -> String {
        let SparseMatmulBwd(SparseMatmul { batch, rows, cols, nnz, .. }) = *self;
        format!("sparse.matmul.bwd<{batch:?}, {rows:?}x{cols:?}, {nnz}>")
    }

    fn inputs(&self) -> Vec<TType> {
        let SparseMatmulBwd(SparseMatmul { dtype, batch, rows, nnz, .. }) = *self;
        vec![TType::new(batch * rows, dtype), TType::new(batch * nnz, DType::I32)]
    }

    fn outputs(&self) -> Vec<TType> {
        let SparseMatmulBwd(SparseMatmul { dtype, rows, cols, .. }) = *self;
        vec![TType::new(rows * cols, dtype)]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
        let SparseMatmulBwd(SparseMatmul { dtype, batch, rows, cols, nnz }) = *self;

        let d = inputs[0];
        let s = inputs[1];
        let o = &mut outputs[0];

        let var_size = (batch * nnz).get_var_size(s.size()).unwrap_or(1);
        let b = batch.evaluate(var_size);

        assert_eq!(s.size(), b * nnz);
        assert_eq!(o.size(), rows * cols);
        assert_eq!(d.size(), b * rows);

        for idx in 0..rows * cols {
            o.write(idx, DValue::zero(dtype));
        }

        for bi in 0..b {
            for ri in 0..rows {
                for ni in 0..nnz {
                    let DValue::I32(idx) = s.read(nnz * bi + ni) else { panic!() };
                    if idx >= 0 && (idx as usize) < cols {
                        let g = d.read(rows * bi + ri);
                        let index = rows * idx as usize + ri;
                        let new_g = CABinary::Add.evaluate(g, o.read(index));
                        o.write(index, new_g.unwrap());
                    }
                }
            }
        }

        true
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(other) = other.downcast::<Self>() { self == other } else { false }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseMatmulBwdMulti(Vec<SparseMatmulBwd>);

impl SparseMatmulBwdMulti {
    pub fn new(bwd: SparseMatmulBwd) -> Self {
        Self(vec![bwd])
    }

    pub fn inner(&self) -> &[SparseMatmulBwd] {
        &self.0
    }

    pub fn push(&mut self, next: SparseMatmulBwd) -> Result<(), IRError> {
        let SparseMatmul { dtype, batch, rows, cols, .. } = self.0[0].0;
        let inner = next.0;

        if inner.dtype != dtype || inner.batch != batch || inner.rows != rows || inner.cols != cols {
            return Err("Mismatched SparseMatmulBwd!".into());
        }

        self.0.push(next);
        Ok(())
    }

    pub fn dtype(&self) -> DType {
        self.0[0].0.dtype
    }

    pub fn batch(&self) -> Size {
        self.0[0].0.batch
    }

    pub fn rows(&self) -> usize {
        self.0[0].0.rows
    }

    pub fn cols(&self) -> usize {
        self.0[0].0.cols
    }
}

impl OpType for SparseMatmulBwdMulti {
    fn opname(&self) -> String {
        let SparseMatmul { batch, rows, cols, .. } = self.0[0].0;
        format!("sparse.matmul.bwd.multi<{batch:?}, {rows:?}x{cols:?}>")
    }

    fn inputs(&self) -> Vec<TType> {
        self.0.iter().flat_map(|x| x.inputs()).collect()
    }

    fn outputs(&self) -> Vec<TType> {
        self.0[0].outputs()
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
        let out = &mut outputs[0];
        for i in 0..out.size() {
            out.write(i, DValue::zero(out.dtype()));
        }

        for (bwd, inps) in self.0.iter().zip(inputs.chunks_exact(2)) {
            let mut this_out = TValue::zeros(out.dtype(), out.size());
            if !bwd.evaluate(inps.to_vec(), vec![&mut this_out]) {
                return false;
            }

            for i in 0..out.size() {
                out.write(i, CABinary::Add.evaluate(out.read(i), this_out.read(i)).unwrap());
            }
        }

        true
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(other) = other.downcast::<Self>() { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate() {
        let lhs = TValue::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let rhs = TValue::I32(vec![0, 1, -1, -1, 2, 2, 1, 0]);
        let mut outputs = TValue::F32(vec![0.0; 4]);

        // [0, 2, 4]   [1, 1]   [2, 10]
        // [1, 3, 5] @ [1, 1] = [4, 14]
        //             [0, 2]
        let matmul = SparseMatmul::new(DType::F32, Size::variable(), 2, 3, 4);
        matmul.evaluate(vec![&lhs, &rhs], vec![&mut outputs]);
        assert_eq!(outputs, TValue::F32(vec![2.0, 4.0, 10.0, 14.0]));
    }

    #[test]
    fn evaluate_bwd() {
        let lhs = TValue::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let rhs = TValue::I32(vec![0, 1, -1, -1, 2, 2, 1, 0]);
        let mut outputs = TValue::F32(vec![0.0; 9]);

        // [0, 3]   [1, 1, 0]   [3, 3,  6]
        // [1, 4] @ [1, 1, 2] = [5, 5,  8]
        // [2, 5]               [7, 7, 10]
        let matmul =
            SparseMatmulBwdMulti::new(SparseMatmulBwd(SparseMatmul::new(DType::F32, Size::variable(), 3, 3, 4)));
        matmul.evaluate(vec![&lhs, &rhs], vec![&mut outputs]);
        assert_eq!(outputs, TValue::F32(vec![3.0, 5.0, 7.0, 3.0, 5.0, 7.0, 6.0, 8.0, 10.0]));
    }
}
