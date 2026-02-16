mod compute;
mod dims;
mod mem;

use std::{
    collections::HashSet,
    fmt::{self, Write},
    rc::Rc,
};

use bullet_compiler::{
    ir::NodeId,
    tensor::{DType, DValue, IRTrace, OpType, Size, TType, TValue, TensorIR, operation::ScalarConstant},
};

pub use compute::ComputeStub;

#[derive(Clone, Debug)]
pub struct Stub {
    pub terminal: Vec<usize>,
    pub inputs: Vec<(TType, bool)>,
    pub outputs: Vec<TType>,
    pub source: String,
}

impl OpType for Stub {
    fn opname(&self) -> String {
        "gpu.stub".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        self.inputs.iter().map(|x| x.0).collect()
    }

    fn outputs(&self) -> Vec<TType> {
        self.outputs.clone()
    }

    fn equals(&self, _: &Rc<dyn OpType>) -> bool {
        false
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        Vec::new()
    }

    fn evaluate(&self, _: Vec<&TValue>, _: Vec<&mut TValue>) {
        unimplemented!()
    }
}

pub struct Codegen {
    size: Size,
    graph: TensorIR,
    tid: NodeId,
    var: NodeId,
    locked: HashSet<NodeId>,
}

impl Codegen {
    pub fn new(size: Size) -> Result<Self, IRTrace> {
        let mut graph = TensorIR::default();

        let [var] = graph.add_op([], Ok::<_, IRTrace>(dims::var_size_stub(size)))?[..] else {
            return Err("Invalid number outputs!".into());
        };

        let [tid] = graph.add_op([var], Ok::<_, IRTrace>(dims::thread_idx_stub(size)))?[..] else {
            return Err("Invalid number outputs!".into());
        };

        Ok(Self { size, graph, tid, var, locked: HashSet::new() })
    }

    pub fn tid(&self) -> NodeId {
        self.tid
    }

    pub fn add_buf(&mut self, ttype: TType) -> NodeId {
        self.graph.add_input(ttype)
    }

    fn add_stub(&mut self, inputs: impl AsRef<[NodeId]>, stub: Stub) -> Result<Vec<NodeId>, IRTrace> {
        for (&(_, is_buf), &id) in stub.inputs.iter().zip(inputs.as_ref()) {
            if self.locked.contains(&id) {
                return Err(format!("Node {id:?} is marked as locked!").into());
            }

            if is_buf != self.graph.is_input(id)? {
                return Err("Cannot read from non-leaf!".into());
            }
        }

        for &terminal in &stub.terminal {
            let id = inputs.as_ref()[terminal];

            if !self.graph.is_input(id)? {
                return Err("Cannot mark non-buffer as terminal!".into());
            }

            self.locked.insert(id);
        }

        self.graph.add_op(inputs, Ok::<_, IRTrace>(stub))
    }

    pub fn add_const(&mut self, value: DValue) -> NodeId {
        self.graph.add_scalar(value, self.size)
    }

    pub fn read(&mut self, buf: NodeId, idx: NodeId) -> Result<NodeId, IRTrace> {
        let buf_ty = self.graph.get_node(buf)?.ty();
        self.add_stub([buf, idx], mem::read_stub(buf_ty, self.size)).map(|x| x[0])
    }

    pub fn write(&mut self, buf: NodeId, val: NodeId) -> Result<(), IRTrace> {
        let buf_ty = self.graph.get_node(buf)?.ty();

        if buf_ty.size() != self.size {
            return Err("Only pointwise writes supported!".into());
        }

        self.add_stub([buf, val, self.tid], mem::write_stub(buf_ty)).map(|_| ())
    }

    pub fn compute(&mut self, inputs: impl AsRef<[NodeId]>, op: ComputeStub) -> Result<Vec<NodeId>, IRTrace> {
        self.add_stub(inputs, op.into())
    }

    pub fn generate(self) -> Result<String, fmt::Error> {
        let name = |id: NodeId| format!("n{}", id.inner());
        let dty = |dtype| match dtype {
            DType::F32 => "float",
            DType::I32 => "int",
        };

        let mut inputs = Vec::new();
        let mut code = String::new();

        for op in self.graph.ordered_operations().unwrap() {
            if op.data().is_input() {
                inputs.push(op.outputs()[0]);
            } else if let Some(scalar) = op.data().downcast::<ScalarConstant>() {
                let nm = name(op.outputs()[0]);
                let s = match scalar.0 {
                    DValue::F32(x) => format!("const float {nm} = {x};"),
                    DValue::I32(x) => format!("const int {nm} = {x};"),
                };
                writeln!(&mut code, "{s}")?;
            } else if let Some(stub) = op.data().downcast::<Stub>() {
                let mut src = stub.source.clone();

                for (i, &id) in op.inputs().iter().enumerate() {
                    src = src.replace(&format!("INPUT{}", i + 1), &name(id));
                }

                for (i, &id) in op.outputs().iter().enumerate() {
                    src = src.replace(&format!("OUTPUT{}", i + 1), &name(id));
                }

                writeln!(&mut code, "{src}")?;
            } else {
                unimplemented!();
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

    use bullet_compiler::tensor::TValue;

    use crate::{
        buffer::Buffer,
        runtime::{Device, Dim3, Gpu, Module, Stream},
    };

    use super::*;

    fn fmadd<G: Gpu>() -> Result<(), G::Error> {
        let threads = Size::variable();
        let ttype = TType::new(threads, DType::F32);

        let mut codegen = Codegen::new(threads).unwrap();
        let input1 = codegen.add_buf(ttype);
        let input2 = codegen.add_buf(ttype);
        let output = codegen.add_buf(ttype);

        let tid = codegen.tid();
        let scalar = codegen.add_const(2.0.into());
        let value1 = codegen.read(input1, tid).unwrap();
        let value2 = codegen.read(input2, tid).unwrap();

        let value3 = codegen.compute([scalar, value1], ComputeStub::binary(ttype, "INPUT1 * INPUT2")).unwrap()[0];
        let value4 = codegen.compute([value3, value2], ComputeStub::binary(ttype, "INPUT1 + INPUT2")).unwrap()[0];
        codegen.write(output, value4).unwrap();

        let source = codegen.generate().unwrap();

        println!("{source}");

        let device = Device::<G>::new(0)?;
        let kernel = Module::new(device.clone(), source)?.get_kernel("kernel")?;
        let stream = Stream::new(device)?;

        let values1 = TValue::F32(vec![1.0, 2.0, 3.0, 4.0]);
        let values2 = TValue::F32(vec![4.0, 3.0, 2.0, 1.0]);

        let input1_buf = Buffer::from_host(stream.clone(), &values1)?.value().0;
        let input2_buf = Buffer::from_host(stream.clone(), &values2)?.value().0;

        fn cast<T>(x: &T) -> *mut c_void {
            (x as *const T).cast_mut().cast()
        }

        unsafe {
            let output_buf = Buffer::uninit(stream.clone(), values1.dtype(), values1.size())?.value();

            let size = values1.size() as i32;
            let input1_ptr = input1_buf.acquire(stream.clone())?.ptr();
            let input2_ptr = input2_buf.acquire(stream.clone())?.ptr();
            let output_ptr = output_buf.clone().acquire(stream.clone())?.ptr();

            let grid_dim = Dim3 { x: 1, y: 1, z: 1 };
            let mut args = [cast(&size), cast(&input1_ptr), cast(&input2_ptr), cast(&output_ptr)];

            kernel.launch(&stream, grid_dim, size as u32, args.as_mut_ptr(), 0)?;

            stream.sync()?;

            let actual = output_buf.to_host(stream.clone())?.value();

            assert_eq!(actual, TValue::F32(vec![6.0, 7.0, 8.0, 9.0]));
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::runtime::cuda::{Cuda, CudaError};

        #[test]
        fn fmadd() -> Result<(), CudaError> {
            super::fmadd::<Cuda>()
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use crate::runtime::rocm::{ROCm, ROCmError};

        #[test]
        fn fmadd() -> Result<(), ROCmError> {
            super::fmadd::<ROCm>()
        }
    }
}
