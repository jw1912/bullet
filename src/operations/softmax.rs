use crate::{
    autograd::Operation,
    tensor::{DenseMatrix, Shape, Tensor},
    ExecutionContext,
};

#[derive(Debug)]
pub struct SoftmaxCrossEntropyLoss;

impl Operation for SoftmaxCrossEntropyLoss {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 && inputs[0] == inputs[1] {
            Ok(Shape::new(1, 1))
        } else {
            Err(format!("Invalid number of inputs in power error! Expected 1, got {}", inputs.len()))
        }
    }

    fn forward(&self, ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        if output.internal.is_empty() {
            output.internal.push((String::from("softmaxed"), DenseMatrix::default()));
            output.internal.push((String::from("individual_losses"), DenseMatrix::default()));
        } else {
            assert_eq!(&output.internal[0].0, "softmaxed");
            assert_eq!(&output.internal[1].0, "individual_losses");
        }

        let (smax, indv) = output.internal.split_at_mut(1);

        DenseMatrix::softmax_crossentropy_loss(
            ctx,
            inputs[0].values.dense(),
            inputs[1].values.dense(),
            output.values.dense_mut(),
            &mut smax[0].1,
            &mut indv[0].1,
        );
    }

    fn backward(&self, _: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        assert_eq!(&output.internal[0].0, "softmaxed");
        assert_eq!(&output.internal[1].0, "individual_losses");

        if let Some(grad) = &mut input1[0].gradients {
            DenseMatrix::backprop_softmax_crossentropy_loss(
                &output.internal[0].1,
                input2[0].values.dense(),
                output.gradients.as_ref().unwrap(),
                grad,
            );
        }

        if let Some(grad) = &mut input2[0].gradients {
            DenseMatrix::backprop_softmax_crossentropy_loss(
                &output.internal[0].1,
                input1[0].values.dense(),
                output.gradients.as_ref().unwrap(),
                grad,
            );
        }
    }
}
