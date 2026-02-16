use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use bullet_compiler::tensor::{DType, DValue, IRTrace, TValue};

use crate::runtime::{BlockOnDrop, BlockResult, Buffer, Device, ReadyToCompileGraph, Stream, TensorInput};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Interpreter;

type Buf = Arc<Mutex<TValue>>;

impl Stream for Interpreter {
    type Device = Self;

    fn block_until_done(&self) -> Result<(), IRTrace> {
        Ok(())
    }

    fn copy_scalars_nonblocking(self: Arc<Self>, copies: impl AsRef<[(DValue, Buf)]>) -> BlockResult<Self, Vec<Buf>> {
        let mut bufs = Vec::new();

        for (val, buf) in copies.as_ref() {
            *buf.lock().map_err(|e| format!("{e:?}"))? = TValue::from(*val);
            bufs.push(buf.clone());
        }

        Ok(BlockOnDrop::new(self, bufs))
    }

    fn make_nonblocking(self: Arc<Self>, src: &TValue) -> Result<BlockOnDrop<Self, (&TValue, Buf)>, IRTrace> {
        let buf = Arc::new(Mutex::new(src.clone()));
        Ok(BlockOnDrop::new(self, (src, buf)))
    }

    fn copy_h2d_nonblocking(
        self: Arc<Self>,
        src: &TValue,
        dst: Buf,
    ) -> Result<BlockOnDrop<Self, (&TValue, Buf)>, IRTrace> {
        *dst.lock().map_err(|e| format!("{e:?}"))? = src.clone();
        Ok(BlockOnDrop::new(self, (src, dst)))
    }

    fn copy_d2h_blocking(self: &Arc<Self>, src: Buf) -> Result<TValue, IRTrace> {
        Ok(src.lock().map_err(|e| format!("{e:?}"))?.clone())
    }

    fn execute_graph(
        self: &Arc<Self>,
        graph: &ReadyToCompileGraph,
        tensors: &HashMap<String, Buf>,
    ) -> BlockResult<Self, Vec<Buf>> {
        let mut inputs = HashMap::new();

        let filtered = tensors.iter().filter(|(name, _)| graph.tensors().get(*name).is_some());

        for (name, tensor) in filtered.clone() {
            let input = graph.tensors().get(name).unwrap();
            let value = tensor.lock().map_err(|e| format!("{e:?}"))?.clone();

            if match input {
                TensorInput::In(input) => inputs.insert(*input, value).is_some(),
                TensorInput::InOut(input, _) => inputs.insert(*input, value).is_some(),
                TensorInput::Out(_) => false,
            } {
                return Err("Input already present!".into());
            }
        }

        let outputs = graph.ir().evaluate(inputs)?;

        for (name, tensor) in filtered {
            let input = graph.tensors().get(name).unwrap();

            if let Some(out) = match input {
                TensorInput::InOut(_, output) => Some(output),
                TensorInput::Out(output) => Some(output),
                TensorInput::In(_) => None,
            } {
                let value = outputs.get(out).ok_or::<IRTrace>("Output node missing!".into())?;
                *tensor.lock().map_err(|e| format!("{e:?}"))? = value.clone();
            }
        }

        Ok(BlockOnDrop::new(self.clone(), tensors.values().cloned().collect()))
    }
}

impl Buffer for Mutex<TValue> {
    type Device = Interpreter;

    fn device(&self) -> Arc<Self::Device> {
        Arc::new(Interpreter)
    }

    fn size(&self) -> usize {
        self.lock().unwrap().size()
    }

    fn dtype(&self) -> DType {
        self.lock().unwrap().dtype()
    }
}

impl Device for Interpreter {
    type Error = IRTrace;
    type Stream = Self;
    type Buffer = Mutex<TValue>;
    type CompiledGraph = ReadyToCompileGraph;

    fn default_stream(self: &Arc<Self>) -> Arc<Self> {
        self.clone()
    }

    fn new_stream(self: &Arc<Self>) -> Result<Arc<Self>, IRTrace> {
        Ok(self.clone())
    }

    fn compile(self: &Arc<Self>, graph: ReadyToCompileGraph) -> Result<ReadyToCompileGraph, IRTrace> {
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bullet_compiler::frontend::{IRBuilder, IRTrace};

    #[test]
    fn test_axby() -> Result<(), IRTrace> {
        let builder = IRBuilder::default();

        let a = builder.add_input(1, DType::F32);
        let b = builder.add_input(1, DType::F32);
        let x = builder.add_input(16, DType::F32);

        let y = ((a.broadcast([1], 0, 16)? * x)? + b.broadcast([1], 0, 16)?)?;

        let ir = builder.build([y]);
        let graph = ReadyToCompileGraph::new(
            ir,
            [
                ("a".to_string(), TensorInput::In(a.node())),
                ("b".to_string(), TensorInput::In(b.node())),
                ("x".to_string(), TensorInput::In(x.node())),
                ("y".to_string(), TensorInput::Out(y.node())),
            ]
            .into(),
        )?;

        let device = Arc::new(Interpreter);

        let compiled = device.compile(graph)?;

        let tensors: HashMap<_, _> = [
            ("a".to_string(), device.make_blocking(&TValue::F32(vec![1.0]))?),
            ("b".to_string(), device.make_blocking(&TValue::F32(vec![1.0]))?),
            ("x".to_string(), device.make_blocking(&TValue::F32(vec![1.0; 16]))?),
            ("y".to_string(), device.make_blocking(&TValue::F32(vec![1.0; 16]))?),
        ]
        .into();

        drop(device.execute_graph(&compiled, &tensors)?);

        let output = tensors.get("y").unwrap().clone();

        assert_eq!(output.lock().unwrap().clone(), TValue::F32(vec![2.0; 16]));

        Ok(())
    }

    #[test]
    fn test_axby_inplace() -> Result<(), IRTrace> {
        let builder = IRBuilder::default();

        let a = builder.add_input(1, DType::F32);
        let b = builder.add_input(1, DType::F32);
        let x = builder.add_input(16, DType::F32);

        let y = ((a.broadcast([1], 0, 16)? * x)? + b.broadcast([1], 0, 16)?)?;

        let ir = builder.build([y]);
        let graph = ReadyToCompileGraph::new(
            ir,
            [
                ("a".to_string(), TensorInput::In(a.node())),
                ("b".to_string(), TensorInput::In(b.node())),
                ("x".to_string(), TensorInput::InOut(x.node(), y.node())),
            ]
            .into(),
        )?;

        let device = Arc::new(Interpreter);

        let compiled = device.compile(graph)?;

        let tensors: HashMap<_, _> = [
            ("a".to_string(), device.make_blocking(&TValue::F32(vec![1.0]))?),
            ("b".to_string(), device.make_blocking(&TValue::F32(vec![1.0]))?),
            ("x".to_string(), device.make_blocking(&TValue::F32(vec![1.0; 16]))?),
        ]
        .into();

        drop(device.execute_graph(&compiled, &tensors)?);

        let output = tensors.get("x").unwrap().clone();

        assert_eq!(output.lock().unwrap().clone(), TValue::F32(vec![2.0; 16]));

        Ok(())
    }
}
