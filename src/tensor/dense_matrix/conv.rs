use crate::backend::{bindings, catch_cudnn, ConvolutionDescription, ExecutionContext};

use super::DenseMatrix;

impl DenseMatrix {
    pub fn convolution_forward(
        ctx: &mut ExecutionContext,
        desc: &ConvolutionDescription,
        filters: &DenseMatrix,
        input: &DenseMatrix,
        output: &mut DenseMatrix,
    ) {
        let alpha = 1f32;
        let beta = 0f32;

        unsafe {
            catch_cudnn(
                bindings::cudnnConvolutionForward(
                    ctx.cudnn,
                    ((&alpha) as *const f32).cast(),
                    desc.input,
                    input.buf.ptr().cast(),
                    desc.filter,
                    filters.buf.ptr().cast(),
                    desc.conv,
                    desc.algo,
                    std::ptr::null_mut(),
                    0,
                    ((&beta) as *const f32).cast(),
                    desc.output,
                    output.buf.mut_ptr().cast(),
                )
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn conv_fwd() {
        
    }
}
