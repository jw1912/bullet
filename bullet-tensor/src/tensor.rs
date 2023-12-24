use std::ffi::c_int;

use crate::{
    bindings::{self, cublasHandle_t, cublasOperation_t, cublasSgemmStridedBatched},
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
    len: usize,
    buf: GpuBuffer,
}

impl TensorBatch {
    /// Creates a new tensor with given `shape` and `len`
    /// length, with a zeroed buffer on the GPU
    pub fn new(shape: Shape, len: usize) -> Self {
        assert!(len > 0, "Cannot have a 0 sized batch!");

        Self {
            shape,
            len,
            buf: GpuBuffer::new(len * shape.size()),
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn len(&self) -> usize {
        self.len
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

    pub fn write_to_cpu(&self, buf: &mut [f32]) {
        self.buf.write_to_cpu(buf);
    }

    /// Single Linear-Transform:
    ///
    /// Computes y[i] = ax[i] on a batch of strided inputs, where
    /// - a is an `m x n` matrix, stored row-major (m columns, n rows).
    /// - x[i] is an `m` dimensional vector.
    /// - y[i] is an `n` dimensional vector
    ///
    /// # Safety
    /// `a` must be initialised, all other sources of unsafety
    /// should trip an assert.
    pub unsafe fn single_lt(handle: cublasHandle_t, a: &Tensor, x: &TensorBatch, y: &TensorBatch) {
        let (m, n) = validate_dims(a.shape(), x, y);

        sgemm(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            n,
            m,
            a.ptr,
            m,
            0,
            x.ptr(),
            m,
            y.ptr(),
            n,
            x.len as c_int,
        );
    }

    /// Single Transposed-Linear-Transform:
    ///
    /// Computes x[i] = (a^T)y[i] on a batch of strided inputs, where
    /// - a is an `m x n` matrix, stored row-major (m columns, n rows).
    /// - x[i] is an `m` dimensional vector.
    /// - y[i] is an `n` dimensional vector
    ///
    /// # Safety
    /// `a` must be initialised, all other sources of unsafety
    /// should trip an assert.
    pub unsafe fn single_tlt(handle: cublasHandle_t, a: &Tensor, y: &TensorBatch, x: &TensorBatch) {
        let (m, n) = validate_dims(a.shape(), x, y);

        sgemm(
            handle,
            cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            a.ptr,
            m,
            0,
            y.ptr(),
            n,
            x.ptr(),
            m,
            x.len as c_int,
        );
    }

    /// Multi Linear-Transform:
    ///
    /// Computes y[i] = a[i]x[i] on a batch of strided inputs, where
    /// - a[i] is an `m x n` matrix, stored row-major (m columns, n rows).
    /// - x[i] is an `m` dimensional vector.
    /// - y[i] is an `n` dimensional vector
    pub fn multi_lt(handle: cublasHandle_t, a: &TensorBatch, x: &TensorBatch, y: &TensorBatch) {
        let (m, n) = validate_dims(a.shape(), x, y);
        assert_eq!(x.len, a.len, "Not all tensor batches are the same length!");

        sgemm(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            n,
            m,
            a.ptr(),
            m,
            a.element_size() as c_int,
            x.ptr(),
            m,
            y.ptr(),
            n,
            x.len as c_int,
        );
    }

    /// Multi Transposed-Linear-Transform:
    ///
    /// Computes x[i] = (a[i]^T)y[i] on a batch of strided inputs, where
    /// - a[i] is an `m x n` matrix, stored row-major (m columns, n rows).
    /// - x[i] is an `m` dimensional vector.
    /// - y[i] is an `n` dimensional vector
    pub fn multi_tlt(handle: cublasHandle_t, a: &TensorBatch, y: &TensorBatch, x: &TensorBatch) {
        let (m, n) = validate_dims(a.shape(), x, y);
        assert_eq!(x.len, a.len, "Not all tensor batches are the same length!");

        sgemm(
            handle,
            cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            a.ptr(),
            m,
            a.element_size() as c_int,
            y.ptr(),
            n,
            x.ptr(),
            m,
            x.len as c_int,
        );
    }

    /// Modifies a batch of tensors in-place.
    fn map(f: unsafe extern "C" fn(usize, *const f32, *mut f32), inp: &Self, out: &Self) {
        assert_eq!(inp.shape(), out.shape(), "Mismatched tensor shapes!");
        assert_eq!(inp.len(), out.len(), "Mismatched batch sizes!");
        let size = inp.num_elements();
        unsafe {
            f(size, inp.ptr(), out.ptr());
        }
    }

    /// Activates a batch of tensors in-place.
    pub fn activate(op: Activation, inp: &Self, out: &Self) {
        match op {
            Activation::ReLU => Self::map(bindings::activateReLU, inp, out),
            Activation::CReLU => Self::map(bindings::activateCReLU, inp, out),
            Activation::SCReLU => Self::map(bindings::activateSCReLU, inp, out),
        }
    }

    /// This calulates `y[i] *= x[i] * op'(op_inv(x[i]))` for a batch of input.
    pub fn backprop_activation(op: Activation, inp: &Self, out: &Self) {
        match op {
            Activation::ReLU => Self::map(bindings::backpropReLU, inp, out),
            Activation::CReLU => Self::map(bindings::backpropCReLU, inp, out),
            Activation::SCReLU => Self::map(bindings::backpropSCReLU, inp, out),
        }
    }
}

fn validate_dims(a_shape: Shape, x: &TensorBatch, y: &TensorBatch) -> (c_int, c_int) {
    assert_eq!(x.shape(), Shape::new(1, a_shape.cols()));
    assert_eq!(y.shape(), Shape::new(1, a_shape.rows()));
    assert_eq!(x.len, y.len, "Not all tensor batches are the same length!");

    (a_shape.cols() as c_int, a_shape.rows() as c_int)
}

#[allow(clippy::too_many_arguments)]
fn sgemm(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    n: c_int,
    m: c_int,
    a_ptr: *const f32,
    a_ld: c_int,
    a_str: c_int,
    x_ptr: *const f32,
    x_ld: c_int,
    y_ptr: *mut f32,
    y_ld: c_int,
    batch_size: c_int,
) {
    let alpha = 1.0;
    let beta = 0.0;

    unsafe {
        cublasSgemmStridedBatched(
            handle,
            transa,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            1,
            m,
            &alpha,
            a_ptr,
            a_ld,
            a_str.into(),
            x_ptr,
            x_ld,
            x_ld.into(),
            &beta,
            y_ptr,
            y_ld,
            y_ld.into(),
            batch_size,
        );
    }
}
