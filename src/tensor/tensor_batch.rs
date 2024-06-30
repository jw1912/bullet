use super::{DeviceBuffer, Shape, Tensor};
use crate::{
    backend::{ops, DeviceHandles},
    Activation,
};

pub struct TensorBatch {
    shape: Shape,
    cap: usize,
    buf: DeviceBuffer,
}

impl TensorBatch {
    /// Creates a new tensor with given `shape` and `cap`
    /// buffer length, with a zeroed buffer on the GPU
    pub fn new(shape: Shape, cap: usize) -> Self {
        assert!(cap > 0, "Cannot have a 0 sized batch!");

        Self { shape, cap, buf: DeviceBuffer::new(cap * shape.size()) }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub(crate) fn ptr(&self) -> *mut f32 {
        self.buf.ptr()
    }

    pub fn element_size(&self) -> usize {
        self.shape.size()
    }

    pub fn num_elements(&self) -> usize {
        self.buf.size()
    }

    pub fn load_from_host(&self, buf: &[f32]) {
        self.buf.load_from_host(buf);
    }

    pub fn write_to_host(&self, buf: &mut [f32]) {
        self.buf.write_to_host(buf);
    }

    pub fn copy_from(&self, other: &Self) {
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.cap(), other.cap());
        self.buf.load_from_device(&other.buf);
    }

    /// # Safety
    /// `a` must be initialised, all other sources of unsafety
    /// should trip an assert.
    pub unsafe fn splat_mul_matrix_vector(
        handle: &DeviceHandles,
        batch_size: usize,
        a: &Tensor,
        x: &TensorBatch,
        y: &TensorBatch,
    ) {
        let (m, n) = validate_dims(a.shape(), x, y);

        ops::splat_mul_matrix_vector(handle, m, n, a.ptr(), x.ptr(), y.ptr(), batch_size);
    }

    /// # Safety
    /// `a` must be initialised, all other sources of unsafety
    /// should trip an assert.
    pub unsafe fn splat_mul_matrixt_vector(
        handle: &DeviceHandles,
        batch_size: usize,
        a: &Tensor,
        y: &TensorBatch,
        x: &TensorBatch,
    ) {
        let (m, n) = validate_dims(a.shape(), x, y);

        ops::splat_mul_matrixt_vector(handle, m, n, a.ptr(), y.ptr(), x.ptr(), batch_size);
    }

    pub fn reduce_add_mul_vector_vectort(
        handle: &DeviceHandles,
        batch_size: usize,
        y: &TensorBatch,
        x: &TensorBatch,
        a: &Tensor,
    ) {
        let a_shape = a.shape();
        assert_eq!(x.shape(), Shape::new(1, a_shape.cols()));
        assert_eq!(y.shape(), Shape::new(1, a_shape.rows()));
        assert_eq!(x.cap(), y.cap(), "Not all tensor caps are the same length!");
        assert!(batch_size <= x.cap());

        unsafe {
            ops::reduce_add_mul_vector_vectort(
                handle,
                a_shape.cols(),
                a_shape.rows(),
                y.ptr(),
                x.ptr(),
                a.ptr(),
                batch_size,
            );
        }
    }

    /// # Safety
    /// `out` must be pointing to valid allocated memory.
    pub unsafe fn reduce_add(
        handle: &DeviceHandles,
        ones: &DeviceBuffer,
        batch_size: usize,
        inp: &TensorBatch,
        out: &Tensor,
    ) {
        assert_eq!(inp.shape(), out.shape());
        ops::reduce_add(handle, ones.ptr(), batch_size, out.num_elements(), inp.ptr(), out.ptr());
    }

    /// # Safety
    /// `inp` must be pointing to valid allocated memory.
    pub unsafe fn splat_add(handle: &DeviceHandles, batch_size: usize, inp: &Tensor, out: &TensorBatch) {
        assert_eq!(inp.shape(), out.shape());
        ops::splat_add(handle, batch_size, out.element_size(), inp.ptr(), out.ptr());
    }

    /// # Safety
    /// `inp` must be pointing to valid allocated memory.
    pub unsafe fn add_to(handle: &DeviceHandles, batch_size: usize, inp: &TensorBatch, out: &TensorBatch) {
        Self::map(ops::add_to, handle, batch_size, inp, out);
    }

    /// Modifies a batch of tensors.
    fn map(
        f: unsafe fn(&DeviceHandles, usize, *const f32, *mut f32),
        handle: &DeviceHandles,
        batch_size: usize,
        inp: &TensorBatch,
        out: &TensorBatch,
    ) {
        assert_eq!(inp.shape(), out.shape(), "Mismatched tensor shapes!");
        assert_eq!(inp.cap(), out.cap(), "Mismatched cap sizes!");
        assert!(batch_size <= inp.cap(), "Overflow!");
        unsafe {
            f(handle, batch_size * inp.element_size(), inp.ptr(), out.ptr());
        }
    }

    /// This calulates `out[i] = op(inp[i])` for a batch of input.
    pub fn activate(handle: &DeviceHandles, batch_size: usize, op: Activation, inp: &TensorBatch, out: &TensorBatch) {
        match op {
            Activation::ReLU => Self::map(ops::activate_relu, handle, batch_size, inp, out),
            Activation::CReLU => Self::map(ops::activate_crelu, handle, batch_size, inp, out),
            Activation::SCReLU => Self::map(ops::activate_screlu, handle, batch_size, inp, out),
        }
    }

    /// This calulates `out[i] = inp[i] * op'(out[i])` for a batch of input.
    pub fn backprop_activation(
        handle: &DeviceHandles,
        batch_size: usize,
        op: Activation,
        inp: &TensorBatch,
        out: &TensorBatch,
    ) {
        match op {
            Activation::ReLU => Self::map(ops::backprop_relu, handle, batch_size, inp, out),
            Activation::CReLU => Self::map(ops::backprop_crelu, handle, batch_size, inp, out),
            Activation::SCReLU => Self::map(ops::backprop_screlu, handle, batch_size, inp, out),
        }
    }

    /// # Safety
    /// `weights` and `biases` must be initialised.
    pub unsafe fn affine(
        handle: &DeviceHandles,
        batch_size: usize,
        weights: &Tensor,
        inputs: &TensorBatch,
        biases: &Tensor,
        outputs: &TensorBatch,
    ) {
        TensorBatch::splat_mul_matrix_vector(handle, batch_size, weights, inputs, outputs);
        TensorBatch::splat_add(handle, batch_size, biases, outputs);
    }

    /// # Safety
    /// `weights` must be initialised.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn backprop_affine(
        handle: &DeviceHandles,
        ones: &DeviceBuffer,
        batch_size: usize,
        weights: &Tensor,
        errors: &TensorBatch,
        inputs: &TensorBatch,
        weights_grad: &Tensor,
        biases_grad: &Tensor,
    ) {
        TensorBatch::reduce_add_mul_vector_vectort(handle, batch_size, errors, inputs, weights_grad);
        TensorBatch::reduce_add(handle, ones, batch_size, errors, biases_grad);
        TensorBatch::splat_mul_matrixt_vector(handle, batch_size, weights, errors, inputs);
    }

    pub fn sigmoid_mpe(
        &self,
        handle: &DeviceHandles,
        batch_size: usize,
        results: &TensorBatch,
        error: &DeviceBuffer,
        power: f32,
    ) {
        assert_eq!(self.shape(), results.shape());
        assert_eq!(self.element_size(), results.element_size());

        unsafe {
            ops::sigmoid_mpe(handle, batch_size, self.ptr(), results.ptr(), error.ptr(), power);
        }
    }

    /// # Safety
    /// `buckets` must be valid.
    pub unsafe fn select(
        handle: &DeviceHandles,
        batch_size: usize,
        buckets: *const u8,
        inp: &TensorBatch,
        out: &TensorBatch,
    ) {
        assert_eq!(inp.element_size() % out.element_size(), 0);

        ops::select(handle, batch_size, inp.element_size(), out.element_size(), buckets, inp.ptr(), out.ptr());
    }

    /// # Safety
    /// `buckets` must be valid.
    pub unsafe fn select_backprop(
        handle: &DeviceHandles,
        batch_size: usize,
        buckets: *const u8,
        inp: &TensorBatch,
        out: &TensorBatch,
    ) {
        assert_eq!(out.element_size() % inp.element_size(), 0);

        out.buf.set_zero();

        ops::select_backprop(handle, batch_size, inp.element_size(), out.element_size(), buckets, inp.ptr(), out.ptr());
    }
}

fn validate_dims(a_shape: Shape, x: &TensorBatch, y: &TensorBatch) -> (usize, usize) {
    assert_eq!(x.shape(), Shape::new(1, a_shape.cols()));
    assert_eq!(y.shape(), Shape::new(1, a_shape.rows()));
    assert_eq!(x.cap(), y.cap(), "Not all tensor caps are the same length!");

    (a_shape.cols(), a_shape.rows())
}
