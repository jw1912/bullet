use std::{fmt, sync::Arc};

use bullet_compiler::graph::{DType, TValue};

pub struct Promise<S: Stream, T>(S, T);

impl<S: Stream, T> Drop for Promise<S, T> {
    fn drop(&mut self) {
        self.0.block_until_done().unwrap();
    }
}

pub struct Tensor<S: Stream>(Arc<S::Buffer>);

impl<S: Stream> Clone for Tensor<S> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

pub trait Stream: Sized {
    type Error: fmt::Debug;
    type Buffer;
    type CompiledGraph;

    fn block_until_done(&self) -> Result<(), Self::Error>;

    fn zeros(&self, size: usize, dtype: DType) -> Result<Self::Buffer, Self::Error>;

    fn async_copy_d2d(&self, src: Tensor<Self>, dst: Tensor<Self>) -> Promise<Self, Tensor<Self>>;

    fn async_copy_h2d(&self, src: TValue, dst: Tensor<Self>) -> Promise<Self, TValue>;

    fn copy_d2h(&self, src: Tensor<Self>) -> TValue;
}

pub trait Runtime {
    type Stream;
}
