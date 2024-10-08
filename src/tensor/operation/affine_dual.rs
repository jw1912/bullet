use crate::tensor::{Activation, Matrix, Shape, SparseMatrix, Tensor};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 4 {
        if inputs[0] * inputs[1] == inputs[3] && inputs[1] == inputs[2] {
            Ok(Shape::new(inputs[3].rows() * 2, inputs[3].cols()))
        } else {
            Err(String::from("Incompatible dims in sparse affine dual!"))
        }
    } else {
        Err(format!("Invalid number of inputs in sparse affine dual! Expected 3, got {}", inputs.len()))
    }
}

pub fn forward(inputs: &[&Tensor], output: &mut Tensor, activation: Activation) {
    let weights = inputs[0].values.dense();
    let biases = inputs[3].values.dense();
    let out = output.values.dense_mut();

    if let (Matrix::Sparse(stm), Matrix::Sparse(ntm)) = (&inputs[1].values, &inputs[2].values) {
        SparseMatrix::affine_dual(weights, stm, ntm, biases, out, activation);
    } else {
        panic!("Inputs must be sparse!");
    }
}

pub fn backprop(output: &Tensor, inputs: &mut [&mut Tensor], activation: Activation) {
    let (input1, inputs2) = inputs.split_at_mut(1);
    let (input2, inputs3) = inputs2.split_at_mut(1);
    let (input3, input4) = inputs3.split_at_mut(1);

    if let (Matrix::Sparse(stm), Matrix::Sparse(ntm)) = (&input2[0].values, &input3[0].values) {
        SparseMatrix::backprop_affine_dual(
            input1[0].values.dense(),
            input1[0].gradients.as_mut().unwrap(),
            stm,
            ntm,
            input4[0].gradients.as_mut().unwrap(),
            output.values.dense(),
            output.gradients.as_ref().unwrap(),
            activation,
        );
    }
}
