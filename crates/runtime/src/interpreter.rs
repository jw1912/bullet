use std::{collections::HashMap, sync::Mutex};

use bullet_compiler::graph::{DType, GraphError, TValue};

use crate::device::{Device, ReadyToCompileGraph, Stream, SyncOnDrop, TensorInput, TensorRef};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Interpreter;

impl Stream for Interpreter {
    type Error = GraphError;
    type Buffer = Mutex<TValue>;
    type CompiledGraph = ReadyToCompileGraph;

    fn block_until_done(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn zeros(&self, size: usize, dtype: DType) -> Result<Self::Buffer, Self::Error> {
        Ok(Mutex::new(TValue::zeros(dtype, size)))
    }

    fn async_copy_h2d(&self, src: TValue, dst: TensorRef<Self>) -> Result<SyncOnDrop<Self, TValue>, Self::Error> {
        *dst.buf().lock().map_err(|e| format!("{e:?}"))? = src;
        Ok(SyncOnDrop::new(*self, TValue::F32(vec![0.0])))
    }

    fn copy_d2h(&self, src: TensorRef<Self>) -> Result<TValue, Self::Error> {
        Ok(src.buf().lock().map_err(|e| format!("{e:?}"))?.clone())
    }

    fn execute(
        &self,
        graph: &Self::CompiledGraph,
        tensors: HashMap<String, TensorRef<Self>>,
    ) -> Result<SyncOnDrop<Self, Vec<TensorRef<Self>>>, Self::Error> {
        let mut inputs = HashMap::new();

        for (name, tensor) in &tensors {
            let input = graph.inputs().get(name).ok_or::<GraphError>("Name not present!".into())?;
            let value = tensor.buf().lock().map_err(|e| format!("{e:?}"))?.clone();

            let err = match input {
                TensorInput::In(input) => inputs.insert(*input, value).is_some(),
                TensorInput::InOut(input, _) => inputs.insert(*input, value).is_some(),
                TensorInput::Out(_) => false,
            };

            if err {
                return Err("Input already present!".into());
            }
        }

        let outputs = graph.graph().evaluate(inputs)?;

        for (name, tensor) in &tensors {
            let input = graph.inputs().get(name).ok_or::<GraphError>("Name not present!".into())?;

            if let Some(out) = match input {
                TensorInput::InOut(_, output) => Some(output),
                TensorInput::Out(output) => Some(output),
                TensorInput::In(_) => None,
            } {
                let value = outputs.get(out).ok_or::<GraphError>("Output node missing!".into())?;
                *tensor.buf().lock().map_err(|e| format!("{e:?}"))? = value.clone();
            }
        }

        Ok(SyncOnDrop::new(*self, tensors.into_values().collect()))
    }
}

impl Device for Interpreter {
    type S = Interpreter;

    fn new_stream(&mut self) -> Result<Self::S, <Self::S as Stream>::Error> {
        Ok(*self)
    }

    fn compile(
        &self,
        graph: ReadyToCompileGraph,
    ) -> Result<<Self::S as Stream>::CompiledGraph, <Self::S as Stream>::Error> {
        Ok(graph)
    }
}
