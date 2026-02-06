use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
};

use bullet_compiler::{
    IR, IRTrace,
    graph::{DType, NodeId, TValue},
};

/// Ensures required objects for nonblocking operations are not
/// dropped before the operation is completed (by syncing)
pub struct BlockOnDrop<S: Stream, T>(S, T);

impl<S: Stream, T> BlockOnDrop<S, T> {
    pub fn new(stream: S, owned: T) -> Self {
        Self(stream, owned)
    }
}

impl<S: Stream, T> Drop for BlockOnDrop<S, T> {
    fn drop(&mut self) {
        self.0.block_until_done().unwrap();
    }
}

/// Atomically reference counted tensor
pub struct TensorRef<S: Stream> {
    buf: Arc<S::Buffer>,
}

impl<S: Stream> TensorRef<S> {
    pub fn buf(&self) -> Arc<S::Buffer> {
        self.buf.clone()
    }
}

impl<S: Stream> PartialEq for TensorRef<S> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buf, &other.buf)
    }
}

impl<S: Stream> Clone for TensorRef<S> {
    fn clone(&self) -> Self {
        Self { buf: self.buf.clone() }
    }
}

pub trait Stream: Clone + Sized {
    type Error: fmt::Debug;
    type Buffer;
    type CompiledGraph;

    fn block_until_done(&self) -> Result<(), Self::Error>;

    fn zeros(&self, size: usize, dtype: DType) -> Result<Self::Buffer, Self::Error>;

    fn copy_h2d_nonblocking<'a>(
        &self,
        src: &'a TValue,
        dst: TensorRef<Self>,
    ) -> Result<BlockOnDrop<Self, &'a TValue>, Self::Error>;

    fn copy_h2d_blocking(&self, src: &TValue, dst: TensorRef<Self>) -> Result<(), Self::Error> {
        drop(self.copy_h2d_nonblocking(src, dst)?);
        Ok(())
    }

    fn copy_d2h_blocking(&self, src: TensorRef<Self>) -> Result<TValue, Self::Error>;

    fn execute(
        &self,
        graph: &Self::CompiledGraph,
        tensors: HashMap<String, TensorRef<Self>>,
    ) -> Result<BlockOnDrop<Self, Vec<TensorRef<Self>>>, Self::Error>;
}

pub enum TensorInput {
    In(NodeId),
    Out(NodeId),
    InOut(NodeId, NodeId),
}

pub trait Device {
    type S: Stream;

    fn default_stream(&self) -> Self::S;

    fn new_stream(&self) -> Result<Self::S, <Self::S as Stream>::Error>;

    fn compile(
        &self,
        graph: ReadyToCompileGraph,
    ) -> Result<<Self::S as Stream>::CompiledGraph, <Self::S as Stream>::Error>;
}

pub struct ReadyToCompileGraph {
    ir: IR,
    tensors: HashMap<String, TensorInput>,
}

impl ReadyToCompileGraph {
    pub fn new(ir: IR, tensors: HashMap<String, TensorInput>) -> Result<ReadyToCompileGraph, IRTrace> {
        let mut present = HashSet::new();

        for value in tensors.values() {
            match value {
                TensorInput::In(input) => {
                    if !present.insert(input) {
                        return Err("Node already accounted for!".into());
                    }

                    if !ir.is_input(*input)? {
                        return Err("Node is not input!".into());
                    }
                }
                TensorInput::Out(output) => {
                    if !present.insert(output) {
                        return Err("Node already accounted for!".into());
                    }

                    if !ir.is_output(*output) {
                        return Err("Node is not output!".into());
                    }
                }
                TensorInput::InOut(input, output) => {
                    if !present.insert(input) || !present.insert(output) {
                        return Err("Node already accounted for!".into());
                    }

                    if !ir.is_input(*input)? {
                        return Err("Node is not input!".into());
                    }

                    if !ir.is_output(*output) {
                        return Err("Node is not output!".into());
                    }

                    if ir.get_node(*input)?.ty() != ir.get_node(*output)?.ty() {
                        return Err("Mismatched node types!".into());
                    }
                }
            }
        }

        Ok(Self { ir, tensors })
    }

    pub fn ir(&self) -> &IR {
        &self.ir
    }

    pub fn tensors(&self) -> &HashMap<String, TensorInput> {
        &self.tensors
    }
}
