pub mod interpreter;

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
};

use bullet_compiler::{
    ir::NodeId,
    tensor::{DType, DValue, IRTrace, TValue, TensorIR},
};

/// Ensures required objects for nonblocking operations are not
/// dropped before the operation is completed (by syncing)
pub struct BlockOnDrop<S: Stream, T>(Arc<S>, T);

impl<S: Stream, T> BlockOnDrop<S, T> {
    pub fn new(stream: Arc<S>, owned: T) -> Self {
        Self(stream, owned)
    }

    pub fn stream(&self) -> Arc<S> {
        self.0.clone()
    }

    pub fn value(&self) -> &T {
        &self.1
    }
}

impl<S: Stream, T> Drop for BlockOnDrop<S, T> {
    fn drop(&mut self) {
        self.0.block_until_done().unwrap();
    }
}

pub trait Buffer {
    type Device;

    fn device(&self) -> Arc<Self::Device>;

    fn size(&self) -> usize;

    fn dtype(&self) -> DType;
}

type Buf<S> = Arc<<<S as Stream>::Device as Device>::Buffer>;
type Err<S> = <<S as Stream>::Device as Device>::Error;
pub type BlockResult<S, T> = Result<BlockOnDrop<S, T>, Err<S>>;

pub trait Stream: Sized {
    type Device: Device<Stream = Self>;

    fn block_until_done(&self) -> Result<(), Err<Self>>;

    fn copy_scalars_nonblocking(
        self: Arc<Self>,
        copies: impl AsRef<[(DValue, Buf<Self>)]>,
    ) -> BlockResult<Self, Vec<Buf<Self>>>;

    fn copy_h2d_nonblocking(self: Arc<Self>, src: &TValue, dst: Buf<Self>) -> BlockResult<Self, (&TValue, Buf<Self>)>;

    fn make_nonblocking(self: Arc<Self>, src: &TValue) -> BlockResult<Self, (&TValue, Buf<Self>)>;

    fn copy_h2d_blocking(self: &Arc<Self>, src: &TValue, dst: Buf<Self>) -> Result<(), Err<Self>> {
        drop(self.clone().copy_h2d_nonblocking(src, dst)?);
        Ok(())
    }

    fn make_blocking(self: &Arc<Self>, src: &TValue) -> Result<Buf<Self>, Err<Self>> {
        self.clone().make_nonblocking(src).map(|block| block.value().1.clone())
    }

    fn copy_d2h_blocking(self: &Arc<Self>, src: Buf<Self>) -> Result<TValue, Err<Self>>;

    fn execute_graph(
        self: &Arc<Self>,
        graph: &<Self::Device as Device>::CompiledGraph,
        tensors: &HashMap<String, Buf<Self>>,
    ) -> BlockResult<Self, Vec<Buf<Self>>>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorInput {
    In(NodeId),
    Out(NodeId),
    InOut(NodeId, NodeId),
}

pub trait Device {
    type Buffer: Buffer<Device = Self>;
    type CompiledGraph;
    type Error: fmt::Debug + From<String>;
    type Stream: Stream<Device = Self>;

    fn default_stream(self: &Arc<Self>) -> Arc<Self::Stream>;

    fn new_stream(self: &Arc<Self>) -> Result<Arc<Self::Stream>, Self::Error>;

    fn compile(self: &Arc<Self>, graph: ReadyToCompileGraph) -> Result<Self::CompiledGraph, Self::Error>;
}

pub struct ReadyToCompileGraph {
    ir: TensorIR,
    tensors: HashMap<String, TensorInput>,
}

impl ReadyToCompileGraph {
    pub fn new(ir: TensorIR, tensors: HashMap<String, TensorInput>) -> Result<ReadyToCompileGraph, IRTrace> {
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

    pub fn ir(&self) -> &TensorIR {
        &self.ir
    }

    pub fn tensors(&self) -> &HashMap<String, TensorInput> {
        &self.tensors
    }
}
