use crate::{bindings, util, Shape, Tensor, TensorBatch};

/// A sparse representation of a tensor with dimensions `(1, input_dim)`.
pub struct SparseTensor {
    cap: usize,
    used: usize,
    chunk_size: usize,
    input_dim: usize,
    max_num_inputs: usize,
    ptr: *mut u16,
}

impl Drop for SparseTensor {
    fn drop(&mut self) {
        unsafe {
            util::free(self.ptr.cast());
        }
    }
}

impl SparseTensor {
    /// # Safety
    /// This creates an uninitialised instance, it is up to the
    /// user to perform an operation which initialises it.
    pub unsafe fn uninit(
        cap: usize,
        input_dim: usize,
        max_num_inputs: usize,
        output_size: usize,
    ) -> Self {
        assert!(input_dim < 65_535, "Unsupported dimension!");
        let chunk_size = determine_chunk_size(output_size);

        Self {
            cap,
            used: 0,
            chunk_size,
            input_dim,
            max_num_inputs,
            ptr: util::malloc(max_num_inputs * cap),
        }
    }

    pub fn clear(&mut self) {
        self.used = 0;
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn append(&mut self, inputs: &[u16]) {
        let num_inputs = inputs.len() / self.max_num_inputs;
        assert!(self.used + num_inputs <= self.cap);

        let used_space = self.used * self.max_num_inputs;

        unsafe {
            util::copy_to_gpu(self.ptr.add(used_space), inputs.as_ptr(), inputs.len());
        }

        self.used += num_inputs;
    }

    /// Sparse Affine Transformation:
    ///
    /// Computes outputs[i] = weights * inputs[i] + biases.
    ///
    /// # Safety
    /// `weights`, `biases` and `inputs` must be initialised properly.
    pub unsafe fn affine(weights: &Tensor, inputs: &Self, biases: &Tensor, outputs: &TensorBatch) {
        assert!(inputs.used > 0);
        let input_dim = inputs.input_dim;
        let output_dim = outputs.element_size();

        assert_eq!(weights.shape(), Shape::new(output_dim, input_dim));
        assert_eq!(biases.shape(), Shape::new(1, output_dim));

        bindings::sparseAffineForward(
            inputs.used,
            inputs.chunk_size,
            inputs.max_num_inputs,
            output_dim,
            weights.ptr(),
            biases.ptr(),
            inputs.ptr,
            outputs.ptr(),
        );
    }

    /// Sparse Affine Transformation:
    ///
    /// Computes backprop for outputs[i] = weights * inputs[i] + biases.
    ///
    /// # Safety
    /// `weights`, `biases` and `errors` must be initialised properly.
    pub unsafe fn affine_backprop(
        weights_grad: &Tensor,
        inputs: &Self,
        biases_grad: &Tensor,
        errors: &TensorBatch,
    ) {
        assert!(inputs.used > 0);
        let input_dim = inputs.input_dim;
        let output_dim = errors.element_size();

        assert_eq!(weights_grad.shape(), Shape::new(output_dim, input_dim));
        assert_eq!(biases_grad.shape(), Shape::new(1, output_dim));

        bindings::sparseAffineBackward(
            inputs.used,
            inputs.chunk_size,
            inputs.max_num_inputs,
            output_dim,
            weights_grad.ptr(),
            biases_grad.ptr(),
            inputs.ptr,
            errors.ptr(),
        );
    }
}

fn determine_chunk_size(mut size: usize) -> usize {
    size -= 1;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    size |= size >> 32;
    size += 1;

    let chunk_size = size / 1024;

    if chunk_size == 0 {
        1
    } else {
        chunk_size
    }
}
