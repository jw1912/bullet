use crate::tensor::{
    DValue, IRTrace, TNode, TType,
    operation::{
        Unary,
        autograd::{CustomAutograd, CustomAutogradOp},
    },
};

#[derive(Debug, PartialEq)]
pub struct FauxQuantise {
    pub ty: TType,
    pub scale: DValue,
    pub zero_point: Option<DValue>,
    pub round: bool,
}

impl FauxQuantise {
    pub fn new(ty: TType, scale: DValue, round: bool) -> Self {
        Self { ty, scale, zero_point: None, round }
    }

    pub fn with_zero_point(mut self, zero_point: DValue) -> Self {
        self.zero_point = Some(zero_point);
        self
    }
}

impl CustomAutograd for FauxQuantise {
    fn opname(&self) -> String {
        "faux-quantise".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.ty]
    }

    fn forward<'a>(&self, inputs: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let [input] = inputs[..] else { return Err("Invalid number of inputs!".into()) };

        let builder = input.builder();
        let size = input.ty().size();
        let scale = builder.scalar(self.scale.clone(), size);
        let op = if self.round { Unary::Round } else { Unary::Truncate };

        // (x - zero_point)
        let centered = match &self.zero_point {
            Some(zp) => {
                let zp = builder.scalar(zp.clone(), size);
                (input - zp)?
            }
            None => input,
        };

        // op((x - zp) * scale) / scale + zero_point
        let quantized = ((scale * centered)?.unary(op)? * scale.unary(Unary::Reciprocal)?)?;
        let out = match &self.zero_point {
            Some(zp) => {
                let zp = builder.scalar(zp.clone(), size);
                (quantized + zp)?
            }
            None => quantized,
        };

        Ok(vec![out])
    }

    fn backward<'a>(&self, _inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        if output_grads.len() != 1 {
            return Err("Invalid number of output grads!".into());
        }

        Ok(output_grads)
    }

    fn equals(&self, other: &CustomAutogradOp) -> bool {
        if let Some(other) = other.downcast::<FauxQuantise>() { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{
        DType, IRTrace, TType, TValue, TensorIR,
        operation::autograd::CustomAutogradOp,
        transform::autograd::{LowerForward, TakeGradient},
        transform::inline::InlineSubgraphs,
    };
    use std::collections::BTreeMap;

    fn evaluate_forward(op: FauxQuantise, input: &[f32]) -> Result<Vec<f32>, IRTrace> {
        let mut ir = TensorIR::default();
        let ty = TType::new(input.len(), DType::F32);
        let input_node = ir.add_input(ty);
        let output_node =
            ir.add_op([input_node], Ok::<CustomAutogradOp, IRTrace>(CustomAutogradOp::new(op)?))?[0];
        ir.transform(LowerForward)?;
        ir.register_output(output_node);
        let mut seeds = BTreeMap::new();
        seeds.insert(input_node, TValue::F32(input.to_vec()));
        let results = ir.evaluate(seeds)?.ok_or("evaluation returned None")?;
        let TValue::F32(out) = results.get(&output_node).cloned().unwrap() else {
            panic!("expected F32 output");
        };
        Ok(out)
    }

    fn evaluate_gradient(op: FauxQuantise, input: &[f32]) -> Result<Vec<f32>, IRTrace> {
        let mut ir = TensorIR::default();
        let ty = TType::new(input.len(), DType::F32);
        let input_node = ir.add_input(ty);
        let output_node =
            ir.add_op([input_node], Ok::<CustomAutogradOp, IRTrace>(CustomAutogradOp::new(op)?))?[0];

        let size = input.len();
        let grad = ir.add_const(TValue::F32(vec![1.0; size]));
        let (transform, grads) = TakeGradient::new(ir.get_parent_op(output_node)?, [grad]);
        ir.transform(transform)?;
        ir.transform(LowerForward)?;
        ir.transform(InlineSubgraphs)?;

        let grad_node = *grads.borrow().get(&input_node).unwrap();
        ir.register_output(grad_node);
        let mut seeds = BTreeMap::new();
        seeds.insert(input_node, TValue::F32(input.to_vec()));
        let results = ir.evaluate(seeds)?.ok_or("evaluation returned None")?;
        let TValue::F32(g) = results.get(&grad_node).cloned().unwrap() else {
            panic!("expected F32 gradient");
        };
        Ok(g)
    }

    #[test]
    fn forward_round() -> Result<(), IRTrace> {
        // scale=10: round(3.4)/10=0.3, round(3.6)/10=0.4, round(-3.4)/10=-0.3
        let op = FauxQuantise::new(TType::new(3, DType::F32), DValue::F32(10.0), true);
        let out = evaluate_forward(op, &[0.34, 0.36, -0.34])?;
        for (got, want) in out.iter().zip([0.3, 0.4, -0.3]) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
        Ok(())
    }

    #[test]
    fn forward_truncate() -> Result<(), IRTrace> {
        // scale=10: truncate(3.9)/10=0.3, truncate(9.0)/10=0.9, truncate(-3.9)/10=-0.3
        let op = FauxQuantise::new(TType::new(3, DType::F32), DValue::F32(10.0), false);
        let out = evaluate_forward(op, &[0.39, 0.90, -0.39])?;
        for (got, want) in out.iter().zip([0.3, 0.9, -0.3]) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
        Ok(())
    }

    #[test]
    fn forward_zero_point() -> Result<(), IRTrace> {
        // scale=2, zp=4: centred=[1,3], *2=[2,6], round=[2,6], /2=[1,3], +4=[5,7]
        let op = FauxQuantise::new(TType::new(2, DType::F32), DValue::F32(2.0), true)
            .with_zero_point(DValue::F32(4.0));
        let out = evaluate_forward(op, &[5.0, 7.0])?;
        for (got, want) in out.iter().zip([5.0, 7.0]) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
        Ok(())
    }

    #[test]
    fn backward_straight_through() -> Result<(), IRTrace> {
        // STE: gradient passes unchanged
        let op = FauxQuantise::new(TType::new(3, DType::F32), DValue::F32(10.0), true);
        let grad = evaluate_gradient(op, &[0.34, 0.36, -0.34])?;
        assert_eq!(grad, vec![1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn opname_inputs_outputs() {
        let op = FauxQuantise::new(TType::new(4, DType::F32), DValue::F32(1.0), true);
        assert_eq!(op.opname(), "faux-quantise");
        assert_eq!(op.inputs(), vec![TType::new(4, DType::F32)]);
    }

    #[test]
    fn equals_works() {
        let a = FauxQuantise::new(TType::new(4, DType::F32), DValue::F32(1.0), true);
        let b = FauxQuantise::new(TType::new(4, DType::F32), DValue::F32(1.0), true);
        let c = FauxQuantise::new(TType::new(4, DType::F32), DValue::F32(2.0), true);
        assert!(a == b);
        assert!(a != c);
    }

    #[test]
    fn with_zero_point_works() {
        let op = FauxQuantise::new(TType::new(4, DType::F32), DValue::F32(1.0), true)
            .with_zero_point(DValue::F32(0.5));
        assert_eq!(op.zero_point, Some(DValue::F32(0.5)));
    }
}
