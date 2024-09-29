use crate::{backend::{ops, ExecutionContext}, Tensor};

use super::{RawTensor, RawTensorValues};

impl RawTensor {
    pub fn add_forward_batched(
        ctx: &ExecutionContext,
        input_a: &Self,
        input_b: &Self,
        output: &mut Self,
    ) {
        assert_eq!(input_a.shape, input_b.shape);
        assert_eq!(input_b.shape, output.shape);

        if let (
            RawTensorValues::DenseFloats(a),
            RawTensorValues::DenseFloats(b),
        ) = (&input_a.values, &input_b.values) {
            match (input_a.len, input_b.len) {
                (1, x) => {
                    output.resize_if_necessary(x);
                    output.len = x;
                    
                    if let RawTensorValues::DenseFloats(o) = &output.values {
                        unsafe {
                            ops::splat_add(
                                ctx,
                                x,
                                input_a.shape.size(),
                                a.ptr(),
                                b.ptr(),
                                o.ptr(),
                            );
                        }
                    }
                }
                (x, 1) => {
                    output.resize_if_necessary(x);
                    output.len = x;
                    
                    if let RawTensorValues::DenseFloats(o) = &output.values {
                        unsafe {
                            ops::splat_add(
                                ctx,
                                x,
                                input_a.shape.size(),
                                b.ptr(),
                                a.ptr(),
                                o.ptr(),
                            );
                        }
                    }
                }
                (x, y) => {
                    if x != y {
                        panic!("Mismatched batch sizes!");
                    }
    
                    unimplemented!();
                }
            }
        }        
    }

    pub fn add_backward_batched(
        ctx: &ExecutionContext,
        inputs: &mut [&mut Tensor],
        output_grad: &Self,
    ) {
        for input in inputs {
            if let Some(grd) = input.gradients.as_mut() {
                backprop_add_single(ctx, &input.values, grd, output_grad);
            }
        }
    }
}

fn backprop_add_single(
    ctx: &ExecutionContext,
    input: &RawTensor, 
    input_grad: &mut RawTensor,
    output_grad: &RawTensor,
) {
    if let RawTensorValues::DenseFloats(out_grd) = &output_grad.values {
        input_grad.resize_if_necessary(input.len);
        input_grad.len = input.len;
        
        if let RawTensorValues::DenseFloats(inp_grd) = &input_grad.values {
            unsafe {
                if input.len == 1 {
                    ops::reduce_add(
                        ctx,
                        output_grad.len,
                        output_grad.shape.size(),
                        out_grd.ptr(),
                        inp_grd.ptr(),
                        true,
                    );
                } else {
                    ops::add_to(
                        ctx,
                        output_grad.size(),
                        out_grd.ptr(),
                        inp_grd.ptr(),
                    );
                }
            }
        }
    }

}
