use crate::tensor::{DenseMatrix, Shape, Tensor};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 2 && inputs[0] == inputs[1] {
        Ok(Shape::new(1, 1))
    } else {
        Err(format!("Invalid number of inputs in power error! Expected 1, got {}", inputs.len()))
    }
}

pub fn forward(power: f32, inputs: &[&Tensor], output: &mut Tensor) {
    DenseMatrix::abs_power_error(power, inputs[0].values.dense(), inputs[1].values.dense(), output.values.dense_mut());
}

pub fn backprop(power: f32, output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, input2) = inputs.split_at_mut(1);

    DenseMatrix::backprop_abs_power_error(
        power,
        input1[0].values.dense(),
        input1[0].gradients.as_mut(),
        input2[0].values.dense(),
        input2[0].gradients.as_mut(),
        output.gradients.as_ref().unwrap(),
    );
}
