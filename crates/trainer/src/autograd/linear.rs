use bullet_compiler::{
    IRTrace,
    operation::{BroadcastAcrossDimension, Matmul, MatrixLayout, ReduceAcrossDimension},
    prelude::IRNode,
};

use super::AutogradOnCoreOp;

impl AutogradOnCoreOp for BroadcastAcrossDimension {
    fn backward<'a>(
        &self,
        _inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
        let op = self.invert().unwrap();
        output_grads[0].builder().add_op([output_grads[0]], op).map(|x| vec![Some(x[0])])
    }
}

impl AutogradOnCoreOp for ReduceAcrossDimension {
    fn backward<'a>(
        &self,
        _inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
        let op = self.invert().unwrap().expect("Reduction backprop only implemented for Sum!");
        output_grads[0].builder().add_op([output_grads[0]], op).map(|x| vec![Some(x[0])])
    }
}

impl AutogradOnCoreOp for Matmul {
    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
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

        Ok(vec![Some(lhs_grad), Some(rhs_grad)])
    }
}
