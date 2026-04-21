use crate::tensor::{
    IRTrace, TNode,
    operation::{ReduceAcrossDimension, Select, SelectPad, SliceAcrossDimension},
};

use super::AutogradOnCoreOp;

impl AutogradOnCoreOp for ReduceAcrossDimension {
    fn backward<'a>(
        &self,
        _inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
        let op = self.invert()?.ok_or::<IRTrace>("Reduction backprop only implemented for Sum!".into())?;
        output_grads[0].builder().add_op([output_grads[0]], op).map(|x| vec![Some(x[0])])
    }
}

impl AutogradOnCoreOp for SliceAcrossDimension {
    fn backward<'a>(
        &self,
        _inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
        let op = self.invert().unwrap();
        output_grads[0].builder().add_op([output_grads[0]], op).map(|x| vec![Some(x[0])])
    }
}

impl AutogradOnCoreOp for Select {
    fn backward<'a>(
        &self,
        inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
        let Select { dtype, batch, inner, divisor } = *self;
        let op = SelectPad { dtype, batch, inner, divisor };
        output_grads[0].builder().add_op([output_grads[0], inputs[1]], op).map(|x| vec![Some(x[0])])
    }
}
