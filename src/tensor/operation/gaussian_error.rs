use crate::tensor::{DenseMatrix, Shape, Tensor};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    todo!("This should go from two inputs to one output.")
}

pub fn forward(inputs: &[&Tensor], output: &mut Tensor) {
    DenseMatrix::gaussian_error(inputs[0].values.dense(), inputs[1].values.dense(), inputs[2].values.dense(), output.values.dense_mut());
}

pub fn backprop(output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, input2) = inputs.split_at_mut(1);
    let (input2, input3) = input2.split_at_mut(1);

    DenseMatrix::backprop_gaussian_error(
        input1[0].values.dense(),
        input1[0].gradients.as_mut(),
        input2[0].values.dense(),
        input2[0].gradients.as_mut(),
        input3[0].values.dense(),
        input3[0].gradients.as_mut(),
        output.gradients.as_ref().unwrap(),
    );
}
