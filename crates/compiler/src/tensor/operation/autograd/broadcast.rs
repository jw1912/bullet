use crate::tensor::{
    IRTrace, TNode,
    operation::{BroadcastAcrossDimension, PadAcrossDimension},
};

use super::AutogradOnCoreOp;

impl AutogradOnCoreOp for BroadcastAcrossDimension {
    fn backward<'a>(
        &self,
        _inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
        let op = self.invert().unwrap();
        output_grads[0].builder().add_op([output_grads[0]], op).map(|x| vec![Some(x[0])])
    }
}

impl AutogradOnCoreOp for PadAcrossDimension {
    fn backward<'a>(
        &self,
        _inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
        let op = self.invert().unwrap();
        output_grads[0].builder().add_op([output_grads[0]], op).map(|x| vec![Some(x[0])])
    }
}
