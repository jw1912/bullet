mod stubs;

use std::fmt::{self, Write};

use bullet_compiler::graph::{DType, Graph, GraphError, NodeId, Size, TType};

pub struct Codegen {
    size: Size,
    graph: Graph,
    tid: NodeId,
    var: NodeId,
}

impl Codegen {
    pub fn new(size: Size) -> Result<Self, GraphError> {
        let mut graph = Graph::default();

        let [var] = graph.add_op([], stubs::VarSizeStub(size))?[..] else {
            return Err("Invalid number outputs!".into());
        };

        let [tid] = graph.add_op([var], stubs::ThreadIdxStub(size))?[..] else {
            return Err("Invalid number outputs!".into());
        };

        Ok(Self { size, graph, tid, var })
    }

    pub fn tid(&self) -> NodeId {
        self.tid
    }

    pub fn add_buf(&mut self, ttype: TType) -> NodeId {
        self.graph.add_input(ttype)
    }

    pub fn read(&mut self, buf: NodeId, idx: NodeId) -> Result<NodeId, GraphError> {
        if !self.graph.is_input(buf) {
            return Err("Cannot read from non-leaf!".into());
        }

        let buf_ty = self.graph.get_node(buf)?.ty();
        let idx_ty = self.graph.get_node(idx)?.ty();

        if idx_ty != TType::new(self.size, DType::I32) {
            return Err(format!("Invalid index type for read: {idx_ty:?}!").into());
        }

        self.graph.add_op([buf, idx], stubs::ReadStub(buf_ty, self.size)).map(|x| x[0])
    }

    pub fn write(&mut self, buf: NodeId, val: NodeId) -> Result<(), GraphError> {
        if !self.graph.is_input(buf) {
            return Err("Cannot write to non-leaf!".into());
        }

        if self.graph.is_input(val) {
            return Err("Cannot write leaf value!".into());
        }

        let buf_ty = self.graph.get_node(buf)?.ty();
        let val_ty = self.graph.get_node(val)?.ty();

        if buf_ty.size() != self.size {
            return Err("Only pointwise writes supported!".into());
        }

        if buf_ty != val_ty {
            return Err(format!("Buffer and value must be same type for write: {buf_ty:?} != {val_ty:?}!").into());
        }

        self.graph.add_op([buf, val, self.tid], stubs::WriteStub(buf_ty)).map(|_| ())
    }

    pub fn compute(&mut self, inputs: impl AsRef<[NodeId]>, op: stubs::ComputeStub) -> Result<Vec<NodeId>, GraphError> {
        for &input in inputs.as_ref() {
            if self.graph.is_input(input) {
                return Err("Cannot compute directly on buffer!".into());
            }
        }

        self.graph.add_op(inputs, op)
    }

    pub fn generate(self) -> Result<String, fmt::Error> {
        let name = |id: NodeId| format!("n{}", id.inner());
        let dty = |dtype| match dtype {
            DType::F32 => "float",
            DType::I32 => "int",
        };

        let mut inputs = Vec::new();
        let mut code = String::new();

        for op_id in self.graph.topo_order_ops().unwrap() {
            let op = self.graph.get_op(op_id).unwrap();

            if op.is_input() {
                inputs.push(op.outputs()[0]);
            } else if op.downcast::<stubs::VarSizeStub>().is_some() {
            } else if op.downcast::<stubs::ThreadIdxStub>().is_some() {
                let tid = name(op.outputs()[0]);
                let mut size = format!("{}", self.size.factor());
                for _ in 0..self.size.var_power() {
                    size += &format!(" * {}", name(self.var));
                }

                writeln!(
                    &mut code,
                    "const int idx_in_grid = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;"
                )?;
                writeln!(&mut code, "const int {tid} = idx_in_grid * blockDim.x + threadIdx.x;")?;
                writeln!(&mut code, "if ({tid} >= ({size})) return;")?;
            } else if let Some(read) = op.downcast::<stubs::ReadStub>() {
                let [buf, idx] = *op.inputs() else { panic!() };
                let [out] = *op.outputs() else { panic!() };

                writeln!(&mut code, "const {} {} = {}[{}];", dty(read.0.dtype()), name(out), name(buf), name(idx),)?;
            } else if op.downcast::<stubs::WriteStub>().is_some() {
                let [buf, val, idx] = *op.inputs() else { panic!() };

                writeln!(&mut code, "{}[{}] = {};", name(buf), name(idx), name(val),)?;
            }
        }

        inputs.sort();

        let mut src = String::new();

        write!(&mut src, "extern \"C\" __global__ void kernel(const int {}", name(self.var)).unwrap();

        for &input in &inputs {
            let dtype = self.graph.get_node(input).unwrap().ty().dtype();
            write!(&mut src, ", {}* {}", dty(dtype), name(input)).unwrap();
        }

        writeln!(&mut src, ") {{").unwrap();

        for line in code.lines() {
            writeln!(&mut src, "    {line}").unwrap();
        }

        writeln!(&mut src, "}}").unwrap();

        Ok(src)
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use bullet_compiler::graph::TValue;

    use crate::{
        buffer::GpuBuffer,
        runtime::{Dim3, Gpu, GpuDevice, GpuModule, GpuStream},
    };

    use super::*;

    fn copy<G: Gpu>() -> Result<(), G::Error> {
        let mut codegen = Codegen::new(Size::variable()).unwrap();

        let ty = TType::new(Size::variable(), DType::F32);
        let input = codegen.add_buf(ty);
        let output = codegen.add_buf(ty);
        let tid = codegen.tid();

        let value = codegen.read(input, tid).unwrap();
        codegen.write(output, value).unwrap();

        let device = GpuDevice::<G>::new(0)?;

        let source = codegen.generate().unwrap();
        let kernel = GpuModule::new(device.clone(), source)?.get_kernel("kernel")?;
        let stream = GpuStream::new(device)?;

        let values = TValue::F32(vec![1.0, 2.0, 3.0, 4.0]);
        let input_buf = GpuBuffer::from_host(stream.clone(), &values)?.value().0;

        fn cast<T>(x: &T) -> *mut c_void {
            (x as *const T).cast_mut().cast()
        }

        unsafe {
            let output_buf = GpuBuffer::uninit(stream.clone(), values.dtype(), values.size())?.value();

            let size = values.size() as i32;
            let input_ptr = input_buf.acquire(stream.clone())?.ptr();
            let output_ptr = output_buf.clone().acquire(stream.clone())?.ptr();

            let grid_dim = Dim3 { x: 1, y: 1, z: 1 };
            let mut args = [cast(&size), cast(&input_ptr), cast(&output_ptr)];

            kernel.launch(&stream, grid_dim, size as u32, args.as_mut_ptr(), 0)?;

            stream.sync()?;

            let actual = output_buf.to_host(stream.clone())?.value();

            assert_eq!(values, actual);
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::runtime::cuda::{Cuda, CudaError};

        #[test]
        fn copy() -> Result<(), CudaError> {
            super::copy::<Cuda>()
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use crate::runtime::rocm::{ROCm, ROCmError};

        #[test]
        fn copy() -> Result<(), ROCmError> {
            super::copy::<ROCm>()
        }
    }
}
