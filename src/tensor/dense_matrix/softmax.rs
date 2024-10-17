use crate::backend::ops;

use super::DenseMatrix;

impl DenseMatrix {
    pub fn softmax_across_columns(input: &Self, output: &mut Self) {
        assert!(input.shape.size() > 0);

        output.reshape_if_needed(input.shape);

        unsafe {
            ops::softmax_across_columns(
                input.shape.rows(),
                input.shape.cols(),
                input.buf.ptr(),
                output.buf.mut_ptr(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::util, tensor::Shape};

    #[test]
    fn softmax() {
        let shape = Shape::new(4, 3);

        let mut input = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        input.load_from_slice(shape, &[2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(input.shape(), shape);

        util::panic_if_device_error("Failed to load data from CPU!");

        DenseMatrix::softmax_across_columns(&input, &mut output);

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, shape);

        let mut buf = [0.0; 12];
        output.write_to_slice(&mut buf);

        assert_eq!(buf, [0.25; 12]);

        util::panic_if_device_error("Failed to write data to CPU!");
    }
}
