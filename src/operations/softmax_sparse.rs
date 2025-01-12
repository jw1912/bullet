use crate::{
    autograd::Operation,
    tensor::{DenseMatrix, ExecutionContext, Matrix, Shape, SparseMatrix, Tensor},
};

#[derive(Debug)]
pub struct SparseSoftmaxCrossEntropyLoss;

impl Operation for SparseSoftmaxCrossEntropyLoss {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 3 && inputs[0] == inputs[1] && inputs[1].cols() == inputs[2].cols() {
            Ok(Shape::new(1, 1))
        } else {
            Err(format!("Invalid shape in sparse softmax: {inputs:?}"))
        }
    }

    fn forward(&self, _: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
        if output.internal.is_empty() {
            output.internal.push((String::from("softmaxed"), DenseMatrix::default()));
            output.internal.push((String::from("individual_losses"), DenseMatrix::default()));
        } else {
            assert_eq!(&output.internal[0].0, "softmaxed");
            assert_eq!(&output.internal[1].0, "individual_losses");
        }

        let (smax, indv) = output.internal.split_at_mut(1);

        let mask = match &inputs[0].values {
            Matrix::Sparse(sparse) => sparse,
            Matrix::Dense(_) => panic!("Dense mask not supported!"),
        };

        SparseMatrix::softmax_crossentropy_loss_masked(
            mask,
            inputs[1].values.dense(),
            inputs[2].values.dense(),
            output.values.dense_mut(),
            &mut smax[0].1,
            &mut indv[0].1,
        );
    }

    fn backward(&self, _: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        assert_eq!(&output.internal[0].0, "softmaxed");
        assert_eq!(&output.internal[1].0, "individual_losses");

        let mask = match &input1[0].values {
            Matrix::Sparse(sparse) => sparse,
            Matrix::Dense(_) => panic!("Dense mask not supported!"),
        };

        if let Some(grad) = &mut input2[0].gradients {
            SparseMatrix::backprop_softmax_crossentropy_loss_masked(
                mask,
                &output.internal[0].1,
                input2[1].values.dense(),
                output.gradients.as_ref().unwrap(),
                grad,
            );
        }

        if let Some(grad) = &mut input2[1].gradients {
            SparseMatrix::backprop_softmax_crossentropy_loss_masked(
                mask,
                &output.internal[0].1,
                input2[0].values.dense(),
                output.gradients.as_ref().unwrap(),
                grad,
            );
        }
    }
}
