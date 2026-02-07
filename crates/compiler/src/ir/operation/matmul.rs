use std::{fmt, rc::Rc};

use crate::ir::{
    graph::{DType, DValue, GraphError, Op, OpType, Size, TType, TValue},
    operation::CABinary,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    pub rows: Size,
    pub cols: Size,
    pub col_mjr: bool,
}

impl MatrixLayout {
    pub fn transpose(&self) -> Self {
        Self { rows: self.cols, cols: self.rows, col_mjr: !self.col_mjr }
    }
}

impl fmt::Debug for MatrixLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MatrixLayout { rows, cols, col_mjr } = *self;
        write!(f, "{rows:?}x{cols:?}.{}", if col_mjr { "C" } else { "R" })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Matmul {
    pub dtype: DType,
    pub batch: Size,
    pub lhs: MatrixLayout,
    pub rhs: MatrixLayout,
}

impl Matmul {
    pub fn new(dtype: DType, batch: impl Into<Size>, lhs: MatrixLayout, rhs: MatrixLayout) -> Result<Self, GraphError> {
        if lhs.cols == rhs.rows {
            Ok(Matmul { dtype, batch: batch.into(), lhs, rhs })
        } else {
            Err(format!("Invalid matmul dims: {:?} != {:?}", lhs.cols, rhs.rows).into())
        }
    }
}

impl OpType for Matmul {
    fn opname(&self) -> String {
        let Matmul { dtype, batch, lhs, rhs } = *self;
        format!("matmul<{dtype:?}, {batch:?}, {lhs:?}, {rhs:?}>")
    }

    fn inputs(&self) -> Vec<TType> {
        let Matmul { dtype, batch, lhs, rhs } = *self;
        vec![TType::new(batch * lhs.rows * lhs.cols, dtype), TType::new(batch * rhs.rows * rhs.cols, dtype)]
    }

    fn outputs(&self) -> Vec<TType> {
        let Matmul { dtype, batch, lhs, rhs } = *self;
        vec![TType::new(batch * lhs.rows * rhs.cols, dtype)]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        let l = inputs[0];
        let r = inputs[1];
        let o = &mut outputs[0];

        let l_var_size = (self.batch * self.lhs.rows * self.lhs.cols).get_var_size(l.size());
        let r_var_size = (self.batch * self.rhs.rows * self.rhs.cols).get_var_size(r.size());

        let var_size = match (l_var_size, r_var_size) {
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
        let m = self.lhs.rows.evaluate(var_size);
        let n = self.lhs.cols.evaluate(var_size);
        let k = self.rhs.cols.evaluate(var_size);

        assert_eq!(b * m * n, l.size());
        assert_eq!(b * n * k, r.size());
        assert_eq!(b * m * k, o.size());

        for bi in 0..b {
            for ki in 0..k {
                for mi in 0..m {
                    let mut sum = DValue::zero(self.dtype);
                    for ni in 0..n {
                        let aidx = m * n * bi + if self.lhs.col_mjr { m * ni + mi } else { n * mi + ni };
                        let bidx = n * k * bi + if self.rhs.col_mjr { n * ki + ni } else { k * ni + ki };
                        let prod = CABinary::Mul.evaluate(l.read(aidx), r.read(bidx)).unwrap();
                        sum = CABinary::Add.evaluate(sum, prod).unwrap();
                    }

                    o.write(m * k * bi + m * ki + mi, sum);
                }
            }
        }
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = Op::downcast_rc::<Self>(other) { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate() {
        let lhs = TValue::I32(vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]);
        let rhs = TValue::I32(vec![5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0]);
        let mut outputs = TValue::I32(vec![0; 8]);
        let var = Size::variable();

        // [0, 2, 4]   [5, 2]   [20, 2]
        // [1, 3, 5] @ [4, 1] = [32, 5]
        //             [3, 0]
        let mut lhs_layout = MatrixLayout { rows: var * 2, cols: var * 3, col_mjr: true };
        let mut rhs_layout = MatrixLayout { rows: var * 3, cols: var * 2, col_mjr: true };
        let matmul = Matmul::new(DType::I32, 2, lhs_layout, rhs_layout).unwrap();
        matmul.evaluate(vec![&lhs, &rhs], vec![&mut outputs]);
        assert_eq!(outputs, TValue::I32(vec![20, 32, 2, 5, 20, 32, 2, 5]));

        // [0, 1, 2]   [5, 4]   [ 5,  2]
        // [3, 4, 5] @ [3, 2] = [32, 20]
        //             [1, 0]
        lhs_layout.col_mjr = false;
        rhs_layout.col_mjr = false;
        let matmul = Matmul::new(DType::I32, 2, lhs_layout, rhs_layout).unwrap();
        matmul.evaluate(vec![&lhs, &rhs], vec![&mut outputs]);
        assert_eq!(outputs, TValue::I32(vec![5, 32, 2, 20, 5, 32, 2, 20]));

        // [0, 2, 4]   [5, 4]   [10,  4]
        // [1, 3, 5] @ [3, 2] = [19, 10]
        //             [1, 0]
        lhs_layout.col_mjr = true;
        rhs_layout.col_mjr = false;
        let matmul = Matmul::new(DType::I32, 2, lhs_layout, rhs_layout).unwrap();
        matmul.evaluate(vec![&lhs, &rhs], vec![&mut outputs]);
        assert_eq!(outputs, TValue::I32(vec![10, 19, 4, 10, 10, 19, 4, 10]));

        // [0, 1, 2]   [5, 2]   [10,  1]
        // [3, 4, 5] @ [4, 1] = [46, 10]
        //             [3, 0]
        lhs_layout.col_mjr = false;
        rhs_layout.col_mjr = true;
        let matmul = Matmul::new(DType::I32, 2, lhs_layout, rhs_layout).unwrap();
        matmul.evaluate(vec![&lhs, &rhs], vec![&mut outputs]);
        assert_eq!(outputs, TValue::I32(vec![10, 46, 1, 10, 10, 46, 1, 10]));
    }
}
