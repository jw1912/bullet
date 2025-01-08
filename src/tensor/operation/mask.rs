use crate::tensor::{Matrix, Shape, SparseMatrix, Tensor};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 2 {
        if inputs[0] == inputs[1] {
            Ok(inputs[0])
        } else {
            Err(format!("Input and mask must have the same shape: {} != {}", inputs[0], inputs[1]))
        }
    } else {
        Err(format!("Invalid number of inputs in linear! Expected 2, got {}", inputs.len()))
    }
}

pub fn forward(inputs: &[&Tensor], output: &mut Tensor) {
    match &inputs[1].values {
        Matrix::Dense(_) => unimplemented!("Masking with dense masks is not supported!"),
        Matrix::Sparse(sparse) => {
            SparseMatrix::mask(inputs[0].values.dense(), sparse, output.values.dense_mut());
        }
    }
}

pub fn backprop(output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, input2) = inputs.split_at_mut(1);

    match &input2[0].values {
        Matrix::Dense(_) => unimplemented!("Masking with dense masks is not supported!"),
        Matrix::Sparse(sparse) => {
            assert!(input2[0].gradients.as_ref().is_none());

            if let Some(grad) = input1[0].gradients.as_mut() {
                SparseMatrix::backprop_mask(output.gradients.as_ref().unwrap(), sparse, grad);
            }
        }
    }
}
