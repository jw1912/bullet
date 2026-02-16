use std::rc::Rc;

use crate::tensor::{DType, DValue, OpType, Size, TType, TValue, TensorOp, operation::CABinary};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseMatmul {
    dtype: DType,
    batch: Size,
    rows: Size,
    cols: Size,
    nnz: usize,
}

impl SparseMatmul {
    pub fn new(dtype: DType, batch: impl Into<Size>, rows: impl Into<Size>, cols: impl Into<Size>, nnz: usize) -> Self {
        SparseMatmul { dtype, batch: batch.into(), rows: rows.into(), cols: cols.into(), nnz }
    }

    pub fn invert(&self) -> SparseMatmulBwd {
        SparseMatmulBwd { dtype: self.dtype, batch: self.batch, rows: self.rows, cols: self.cols, nnz: self.nnz }
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

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        let d = inputs[0];
        let s = inputs[1];
        let o = &mut outputs[0];

        let d_var_size = (self.rows * self.cols).get_var_size(d.size());
        let s_var_size = (self.batch * self.nnz).get_var_size(s.size());

        let var_size = match (d_var_size, s_var_size) {
            (None, None) => 1,
            (Some(x), None) => x,
            (None, Some(x)) => x,
            (Some(x), Some(y)) => {
                if x == y {
                    x
                } else {
                    panic!()
                }
            }
        };

        let b = self.batch.evaluate(var_size);
        let r = self.rows.evaluate(var_size);
        let c = self.cols.evaluate(var_size);

        assert_eq!(s.size(), b * self.nnz);
        assert_eq!(d.size(), r * c);
        assert_eq!(o.size(), b * r);

        for bi in 0..b {
            for ri in 0..r {
                let mut sum = DValue::zero(self.dtype);

                for ni in 0..self.nnz {
                    let DValue::I32(idx) = s.read(self.nnz * bi + ni) else { panic!() };
                    if idx >= 0 && (idx as usize) < c {
                        sum = CABinary::Add.evaluate(sum, d.read(r * idx as usize + ri)).unwrap();
                    }
                }

                o.write(r * bi + ri, sum);
            }
        }
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = TensorOp::downcast_rc::<Self>(other) { self == other } else { false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseMatmulBwd {
    dtype: DType,
    batch: Size,
    rows: Size,
    cols: Size,
    nnz: usize,
}

impl SparseMatmulBwd {
    pub fn new(dtype: DType, batch: impl Into<Size>, rows: impl Into<Size>, cols: impl Into<Size>, nnz: usize) -> Self {
        SparseMatmulBwd { dtype, batch: batch.into(), rows: rows.into(), cols: cols.into(), nnz }
    }
}

impl OpType for SparseMatmulBwd {
    fn opname(&self) -> String {
        let SparseMatmulBwd { batch, rows, cols, nnz, .. } = *self;
        format!("sparse.matmul.bwd<{batch:?}, {rows:?}x{cols:?}, {nnz}>")
    }

    fn inputs(&self) -> Vec<TType> {
        let SparseMatmulBwd { dtype, batch, rows, nnz, .. } = *self;
        vec![TType::new(batch * rows, dtype), TType::new(batch * nnz, DType::I32)]
    }

    fn outputs(&self) -> Vec<TType> {
        let SparseMatmulBwd { dtype, rows, cols, .. } = *self;
        vec![TType::new(rows * cols, dtype)]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        let d = inputs[0];
        let s = inputs[1];
        let o = &mut outputs[0];

        let o_var_size = (self.rows * self.cols).get_var_size(o.size());
        let s_var_size = (self.batch * self.nnz).get_var_size(s.size());

        let var_size = match (o_var_size, s_var_size) {
            (None, None) => 1,
            (Some(x), None) => x,
            (None, Some(x)) => x,
            (Some(x), Some(y)) => {
                if x == y {
                    x
                } else {
                    panic!()
                }
            }
        };

        let b = self.batch.evaluate(var_size);
        let r = self.rows.evaluate(var_size);
        let c = self.cols.evaluate(var_size);

        assert_eq!(s.size(), b * self.nnz);
        assert_eq!(o.size(), r * c);
        assert_eq!(d.size(), b * r);

        for idx in 0..r * c {
            o.write(idx, DValue::zero(self.dtype));
        }

        for bi in 0..b {
            for ri in 0..r {
                for ni in 0..self.nnz {
                    let DValue::I32(idx) = s.read(self.nnz * bi + ni) else { panic!() };
                    if idx >= 0 && (idx as usize) < c {
                        let g = d.read(r * bi + ri);
                        let index = r * idx as usize + ri;
                        let new_g = CABinary::Add.evaluate(g, o.read(index));
                        o.write(index, new_g.unwrap());
                    }
                }
            }
        }
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = TensorOp::downcast_rc::<Self>(other) { self == other } else { false }
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
        let matmul = SparseMatmulBwd::new(DType::F32, Size::variable(), 3, 3, 4);
        matmul.evaluate(vec![&lhs, &rhs], vec![&mut outputs]);
        assert_eq!(outputs, TValue::F32(vec![3.0, 5.0, 7.0, 3.0, 5.0, 7.0, 6.0, 8.0, 10.0]));
    }
}
