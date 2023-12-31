use std::ffi::c_int;

use crate::{
    bindings::{self, cublasHandle_t, cublasOperation_t},
    util, Activation, GpuBuffer, Shape,
};

/// Single Rank-2 Tensor on the GPU.
/// This data type does not own the memory it points to,
/// it must be manually allocated and freed, or used to
/// borrow data only.
pub struct Tensor {
    shape: Shape,
    ptr: *mut f32,
}

impl Tensor {
    /// # Safety
    /// This creates an uninitialised instance, it is up to the
    /// user to perform an operation which initialises it.
    pub unsafe fn uninit(shape: Shape) -> Self {
        Self {
            shape,
            ptr: std::ptr::null_mut(),
        }
    }

    /// # Safety
    /// You can set this to point to anywhere.
    pub unsafe fn set_ptr(&mut self, ptr: *mut f32) {
        self.ptr = ptr;
    }

    pub fn calloc(&mut self) {
        self.ptr = util::calloc(self.num_elements());
    }

    /// # Safety
    /// Don't double free.
    pub unsafe fn free(&mut self) {
        util::free(self.ptr.cast());
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn ptr(&self) -> *mut f32 {
        self.ptr
    }

    pub fn reshape(&mut self, cols: usize, rows: usize) {
        self.shape.reshape(cols, rows);
    }

    pub fn num_elements(&self) -> usize {
        self.shape.size()
    }

    pub fn load_from_cpu(&self, buf: &[f32]) {
        assert!(
            !self.ptr.is_null(),
            "Attempting to dereference null pointer!"
        );

        assert!(
            buf.len() == self.num_elements(),
            "Must be exactly the same size!"
        );

        util::copy_to_gpu(self.ptr, buf.as_ptr(), buf.len());
    }

    pub fn write_to_cpu(&self, buf: &mut [f32]) {
        assert!(
            !self.ptr.is_null(),
            "Attempting to dereference null pointer!"
        );

        assert!(
            buf.len() == self.num_elements(),
            "Must be exactly the same size!"
        );

        util::copy_from_gpu(buf.as_mut_ptr(), self.ptr, buf.len());
    }
}

pub struct TensorBatch {
    shape: Shape,
    cap: usize,
    buf: GpuBuffer,
}

impl TensorBatch {
    /// Creates a new tensor with given `shape` and `cap`
    /// buffer length, with a zeroed buffer on the GPU
    pub fn new(shape: Shape, cap: usize) -> Self {
        assert!(cap > 0, "Cannot have a 0 sized batch!");

        Self {
            shape,
            cap,
            buf: GpuBuffer::new(cap * shape.size()),
        }
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

    pub fn load_from_cpu(&self, buf: &[f32]) {
        self.buf.load_from_cpu(buf);
    }

    pub fn offset_load_from_cpu(&self, buf: &[f32], offset: usize) {
        self.buf.offset_load_from_cpu(buf, offset);
    }

    pub fn write_to_cpu(&self, buf: &mut [f32]) {
        self.buf.write_to_cpu(buf);
    }

    /// # Safety
    /// `a` must be initialised, all other sources of unsafety
    /// should trip an assert.
    pub unsafe fn splat_lt_nn(
        handle: cublasHandle_t,
        batch_size: usize,
        a: &Tensor,
        x: &TensorBatch,
        y: &TensorBatch,
    ) {
        let (m, n) = validate_dims(a.shape(), x, y);

        sgemv::<false>(
            handle,
            m,
            n,
            a.ptr,
            0,
            x.ptr(),
            y.ptr(),
            batch_size as c_int,
        );
    }

    /// # Safety
    /// `a` must be initialised, all other sources of unsafety
    /// should trip an assert.
    pub unsafe fn splat_lt_tn(
        handle: cublasHandle_t,
        batch_size: usize,
        a: &Tensor,
        y: &TensorBatch,
        x: &TensorBatch,
    ) {
        let (m, n) = validate_dims(a.shape(), x, y);

        sgemv::<true>(
            handle,
            m,
            n,
            a.ptr,
            0,
            y.ptr(),
            x.ptr(),
            batch_size as c_int,
        );
    }

    pub fn lt_nn(
        handle: cublasHandle_t,
        batch_size: usize,
        a: &TensorBatch,
        x: &TensorBatch,
        y: &TensorBatch,
    ) {
        let (m, n) = validate_dims(a.shape(), x, y);
        assert_eq!(x.cap(), a.cap(), "Not all tensor caps are the same length!");

        sgemv::<false>(
            handle,
            m,
            n,
            a.ptr(),
            a.element_size() as c_int,
            x.ptr(),
            y.ptr(),
            batch_size as c_int,
        );
    }

    pub fn lt_tn(
        handle: cublasHandle_t,
        batch_size: usize,
        a: &TensorBatch,
        y: &TensorBatch,
        x: &TensorBatch,
    ) {
        let (m, n) = validate_dims(a.shape(), x, y);
        assert_eq!(x.cap(), a.cap(), "Not all tensor caps are the same length!");

        sgemv::<true>(
            handle,
            m,
            n,
            a.ptr(),
            a.element_size() as c_int,
            y.ptr(),
            x.ptr(),
            batch_size as c_int,
        );
    }

    pub fn lt_nt(
        handle: cublasHandle_t,
        batch_size: usize,
        y: &TensorBatch,
        x: &TensorBatch,
        a: &TensorBatch,
    ) {
        let a_shape = a.shape();
        assert_eq!(x.shape(), Shape::new(1, a_shape.cols()));
        assert_eq!(y.shape(), Shape::new(1, a_shape.rows()));
        assert_eq!(x.cap(), y.cap(), "Not all tensor caps are the same length!");
        assert_eq!(x.cap(), a.cap(), "Not all tensor caps are the same length!");

        let m = a_shape.cols() as c_int;
        let n = a_shape.rows() as c_int;

        sgemm(
            handle,
            m,
            n,
            y.ptr(),
            x.ptr(),
            a.ptr(),
            a.element_size() as c_int,
            batch_size as c_int,
        );
    }

    /// Modifies a batch of tensors.
    fn map(
        f: unsafe extern "C" fn(usize, *const f32, *mut f32),
        batch_size: usize,
        inp: &Self,
        out: &Self,
    ) {
        assert_eq!(inp.shape(), out.shape(), "Mismatched tensor shapes!");
        assert_eq!(inp.cap(), out.cap(), "Mismatched cap sizes!");
        assert!(batch_size <= inp.cap(), "Overflow!");
        unsafe {
            f(batch_size * inp.element_size(), inp.ptr(), out.ptr());
        }
    }

    /// This calulates `out[i] = op(inp[i])` for a batch of input.
    pub fn activate(batch_size: usize, op: Activation, inp: &Self, out: &Self) {
        match op {
            Activation::ReLU => Self::map(bindings::activateReLU, batch_size, inp, out),
            Activation::CReLU => Self::map(bindings::activateCReLU, batch_size, inp, out),
            Activation::SCReLU => Self::map(bindings::activateSCReLU, batch_size, inp, out),
        }
    }

    /// This calulates `out[i] = inp[i] * op'(out[i])` for a batch of input.
    pub fn backprop_activation(batch_size: usize, op: Activation, inp: &Self, out: &Self) {
        match op {
            Activation::ReLU => Self::map(bindings::backpropReLU, batch_size, inp, out),
            Activation::CReLU => Self::map(bindings::backpropCReLU, batch_size, inp, out),
            Activation::SCReLU => Self::map(bindings::backpropSCReLU, batch_size, inp, out),
        }
    }

    /// # Safety
    /// `weights` must be initialised.
    pub unsafe fn affine(
        handle: cublasHandle_t,
        batch_size: usize,
        weights: &Tensor,
        inputs: &TensorBatch,
        biases: &Tensor,
        outputs: &TensorBatch,
    ) {
        TensorBatch::splat_lt_nn(handle, batch_size, weights, inputs, outputs);
        TensorBatch::splat_add(batch_size, biases, outputs);
    }

    /// # Safety
    /// `weights` must be initialised.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn backprop_affine(
        handle: cublasHandle_t,
        ones: &GpuBuffer,
        batch_size: usize,
        weights: &Tensor,
        errors: &TensorBatch,
        inputs: &TensorBatch,
        weights_grad: &Tensor,
        biases_grad: &Tensor,
        weights_intermediate: &TensorBatch,
    ) {
        TensorBatch::lt_nt(handle, batch_size, errors, inputs, weights_intermediate);
        TensorBatch::reduce_add(handle, ones, batch_size, weights_intermediate, weights_grad);
        TensorBatch::reduce_add(handle, ones, batch_size, errors, biases_grad);
        TensorBatch::splat_lt_tn(handle, batch_size, weights, errors, inputs);
    }

    /// # Safety
    /// `out` must be pointing to valid allocated memory.
    pub unsafe fn reduce_add(handle: cublasHandle_t, ones: &GpuBuffer, batch_size: usize, inp: &TensorBatch, out: &Tensor) {
        assert_eq!(inp.shape(), out.shape());
        let alpha = 1.0;
        let beta = 0.0;

        let m = batch_size as c_int;
        let n = out.num_elements() as c_int;

        bindings::cublasSgemv_v2(
            handle,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            &alpha,
            inp.ptr(),
            n,
            ones.ptr(),
            0,
            &beta,
            out.ptr(),
            1,
        );
    }

    /// # Safety
    /// `inp` must be pointing to valid allocated memory.
    pub unsafe fn splat_add(batch_size: usize, inp: &Tensor, out: &TensorBatch) {
        assert_eq!(inp.shape(), out.shape());
        bindings::splatAdd(batch_size, out.element_size(), inp.ptr(), out.ptr());
    }

    pub fn sigmoid_mse(&self, batch_size: usize, results: &TensorBatch, error: &GpuBuffer) {
        assert_eq!(error.size(), 1);
        assert_eq!(self.shape(), results.shape());
        assert_eq!(self.element_size(), results.element_size());

        unsafe {
            bindings::sigmoidMSE(batch_size, self.ptr(), results.ptr(), error.ptr());
        }
    }
}

fn validate_dims(a_shape: Shape, x: &TensorBatch, y: &TensorBatch) -> (c_int, c_int) {
    assert_eq!(x.shape(), Shape::new(1, a_shape.cols()));
    assert_eq!(y.shape(), Shape::new(1, a_shape.rows()));
    assert_eq!(x.cap(), y.cap(), "Not all tensor caps are the same length!");

    (a_shape.cols() as c_int, a_shape.rows() as c_int)
}

#[allow(clippy::too_many_arguments)]
fn sgemv<const TRANSA: bool>(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    a_ptr: *const f32,
    a_str: c_int,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    batch_size: c_int,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let (transa, x_ld, y_ld) = if TRANSA {
        (cublasOperation_t::CUBLAS_OP_T, n, m)
    } else {
        (cublasOperation_t::CUBLAS_OP_N, m, n)
    };

    unsafe {
        bindings::cublasSgemvStridedBatched(
            handle,
            transa,
            n, m,
            &alpha,
            a_ptr,
            n,
            a_str.into(),
            x_ptr,
            1,
            x_ld.into(),
            &beta,
            y_ptr,
            1,
            y_ld.into(),
            batch_size,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn sgemm(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    y_ptr: *const f32,
    x_ptr: *const f32,
    a_ptr: *mut f32,
    a_str: c_int,
    batch_size: c_int,
) {
    let alpha = 1.0;
    let beta = 0.0;

    unsafe {
        bindings::cublasSgemmStridedBatched(
            handle,
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_T,
            n,
            m,
            1,
            &alpha,
            y_ptr,
            n,
            n.into(),
            x_ptr,
            m,
            m.into(),
            &beta,
            a_ptr,
            n,
            a_str.into(),
            batch_size,
        );
    }
}
