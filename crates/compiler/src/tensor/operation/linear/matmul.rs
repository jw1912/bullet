use std::fmt;

use crate::{
    ir::IRError,
    tensor::{DType, DValue, IRTrace, OpType, Size, TNode, TType, TValue, TensorOp, operation::CABinary},
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
    pub fn new(dtype: DType, batch: impl Into<Size>, lhs: MatrixLayout, rhs: MatrixLayout) -> Result<Self, IRError> {
        if lhs.cols == rhs.rows {
            Ok(Matmul { dtype, batch: batch.into(), lhs, rhs })
        } else {
            Err(format!("Invalid matmul dims: {:?} != {:?}", lhs.cols, rhs.rows).into())
        }
    }
}

impl OpType for Matmul {
    fn opname(&self) -> String {
        let Matmul { batch, lhs, rhs, .. } = *self;
        format!("matmul<{batch:?}, {lhs:?}, {rhs:?}>")
    }

    fn inputs(&self) -> Vec<TType> {
        let Matmul { dtype, batch, lhs, rhs } = *self;
        vec![TType::new(batch * lhs.rows * lhs.cols, dtype), TType::new(batch * rhs.rows * rhs.cols, dtype)]
    }

    fn outputs(&self) -> Vec<TType> {
        let Matmul { dtype, batch, lhs, rhs } = *self;
        vec![TType::new(batch * lhs.rows * rhs.cols, dtype)]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
        let l = inputs[0];
        let r = inputs[1];
        let o = &mut outputs[0];

        let b = self.batch.get();
        let m = self.lhs.rows.get();
        let n = self.lhs.cols.get();
        let k = self.rhs.cols.get();

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

        true
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(other) = other.downcast::<Self>() { self == other } else { false }
    }

    fn backward<'a>(&self, inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let Matmul { dtype, batch, lhs, rhs } = *self;
        let grad = MatrixLayout { col_mjr: true, rows: lhs.rows, cols: rhs.cols };

        let lhs_node = inputs[0];
        let rhs_node = inputs[1];
        let grad_node = output_grads[0];

        let builder = grad_node.builder();

        let lhs_grad = if lhs.col_mjr {
            let op = Matmul::new(dtype, batch, grad, rhs.transpose())?;
            builder.add_op([grad_node, rhs_node], op)?[0]
        } else {
            let op = Matmul::new(dtype, batch, rhs, grad.transpose())?;
            builder.add_op([rhs_node, grad_node], op)?[0]
        };

        let rhs_grad = if rhs.col_mjr {
            let op = Matmul::new(dtype, batch, lhs.transpose(), grad)?;
            builder.add_op([lhs_node, grad_node], op)?[0]
        } else {
            let op = Matmul::new(dtype, batch, grad.transpose(), lhs)?;
            builder.add_op([grad_node, lhs_node], op)?[0]
        };

        Ok(vec![lhs_grad, rhs_grad])
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

        // [0, 2, 4]   [5, 2]   [20, 2]
        // [1, 3, 5] @ [4, 1] = [32, 5]
        //             [3, 0]
        let mut lhs_layout = MatrixLayout { rows: 2.into(), cols: 3.into(), col_mjr: true };
        let mut rhs_layout = MatrixLayout { rows: 3.into(), cols: 2.into(), col_mjr: true };
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
