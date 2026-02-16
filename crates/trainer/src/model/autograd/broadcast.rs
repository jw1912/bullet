use bullet_compiler::{
    frontend::{IRNode, IRTrace},
    tensor::operation::{BroadcastAcrossDimension, PadAcrossDimension},
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

impl AutogradOnCoreOp for PadAcrossDimension {
    fn backward<'a>(
        &self,
        _inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
        let op = self.invert().unwrap();
        output_grads[0].builder().add_op([output_grads[0]], op).map(|x| vec![Some(x[0])])
    }
}
