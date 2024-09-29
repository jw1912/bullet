use crate::backend::{ops, ExecutionContext};

use super::DenseTensor;

impl DenseTensor {
    pub fn add(
        ctx: &mut ExecutionContext,
        input_a: &Self,
        input_b: &Self,
        output: &mut Self,
    ) {
        assert_eq!(input_a.shape.rows(), input_b.shape.rows());

        if input_a.shape == input_b.shape {
            output.reshape_if_needed(input_a.shape);
            unsafe {
                ops::add_matrices(
                    ctx,
                    output.shape.rows(),
                    output.shape.cols(),
                    input_a.buf.ptr(),
                    input_b.buf.ptr(),
                    output.buf.ptr(),
                );
            }
        } else if input_a.shape.cols() == 1 {
            output.reshape_if_needed(input_b.shape);
            unsafe {
                ops::splat_add(
                    ctx,
                    input_b.shape.cols(),
                    input_b.shape.rows(),
                    input_a.buf.ptr(),
                    input_b.buf.ptr(),
                    output.buf.ptr(),
                );
            }
        } else if input_b.shape.cols() == 1 {
            output.reshape_if_needed(input_a.shape);
            unsafe {
                ops::splat_add(
                    ctx,
                    input_a.shape.cols(),
                    input_a.shape.rows(),
                    input_b.buf.ptr(),
                    input_a.buf.ptr(),
                    output.buf.ptr(),
                );
            }
        } else {
            panic!("Invalid shape pairs!")
        };
    }

    pub fn add_backward(
        ctx: &mut ExecutionContext,
        input_a: &Self,
        input_a_grad: Option<&mut Self>,
        input_b: &Self,
        input_b_grad: Option<&mut Self>,
        output_grad: &Self,
    ) {
        if let Some(grd) = input_a_grad {
            backprop_add_single(ctx, input_a, grd, output_grad);
        }

        if let Some(grd) = input_b_grad {
            backprop_add_single(ctx, input_b, grd, output_grad);
        }
    }
}

fn backprop_add_single(
    ctx: &mut ExecutionContext,
    input: &DenseTensor, 
    input_grad: &mut DenseTensor,
    output_grad: &DenseTensor,
) {
    input_grad.reshape_if_needed(input.shape);
    if input.shape.cols() == output_grad.shape.cols() {
        unsafe {
            ops::add_matrix_to(
                ctx,
                output_grad.shape.cols(),
                output_grad.shape.rows(),
                output_grad.buf.ptr(),
                input_grad.buf.ptr(),
            );
        }
    } else if input.shape.cols() == 1 {
        println!("reduce adding!");
        unsafe {
            ops::reduce_add_cols(
                ctx,
                output_grad.shape.cols(),
                output_grad.shape.rows(),
                output_grad.buf.ptr(),
                input_grad.buf.ptr(),
                true,
            );
        }
    } else {
        panic!("Invalid shape pairs!")
    };
}

#[cfg(test)]
mod tests {
    use crate::tensor::Shape;
    use super::*;

    #[test]
    fn add() {
        let mut ctx = ExecutionContext::default();

        let shape1 = Shape::new(3, 3);
        let shape2 = Shape::new(3, 1);

        let mut input1 = DenseTensor::default();
        let mut input2 = DenseTensor::default();
        let mut input1_grad = DenseTensor::default();
        let mut input2_grad = DenseTensor::default();
        let mut output = DenseTensor::default();

        // load tensors from CPU
        {
            input1.load_from_slice(
                shape1,
                &[
                    -1.0, 4.0, 2.0,
                    -2.0, 0.0, -3.0,
                    1.0, 1.0, 1.0,
                ],
            );

            input2.load_from_slice(
                shape2,
                &[1.0, 2.0, 3.0],
            );

            assert_eq!(input1.shape(), shape1);
            assert_eq!(input2.shape(), shape2);
        }

        // add
        {
            DenseTensor::add(
                &mut ctx,
                &input1,
                &input2,
                &mut output,
            );

            assert_eq!(output.shape(), Shape::new(3, 3));

            let mut buf = [0.0; 9];
            output.write_to_slice(&mut buf);
            assert_eq!(
                buf,
                [
                    0.0, 6.0, 5.0,
                    -1.0, 2.0, 0.0,
                    2.0, 3.0, 4.0,
                ],
            );
        }

        // backprop add
        {    
            DenseTensor::add_backward(
                &mut ctx,
                &input1,
                Some(&mut input1_grad),
                &input2,
                Some(&mut input2_grad),
                &output,
            );

            assert_eq!(input1_grad.shape(), shape1);
            assert_eq!(input2_grad.shape(), shape2);

            let mut grad1 = [0.0; 9];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(
                grad1,
                [
                    0.0, 6.0, 5.0,
                    -1.0, 2.0, 0.0,
                    2.0, 3.0, 4.0,
                ]
            );

            let mut grad2 = [0.0; 3];
            input2_grad.write_to_slice(&mut grad2);
            assert_eq!(grad2, [1.0, 11.0, 9.0]);
        }
    }
}
