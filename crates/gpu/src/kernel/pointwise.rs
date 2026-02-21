mod op;
mod str;

use std::{
    collections::HashSet,
    fmt::{self, Write},
    rc::Rc,
};

use bullet_compiler::{
    ir::{IR, IRError, NodeId, TypeSystem},
    tensor::{
        DType, DValue, Size, TType,
        operation::{CABinary, Unary},
    },
};

use op::{MemIO, PointwiseOp};

use crate::{kernel::KernelSrc, runtime::Dim3};

use self::str::{code_str, tystr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PType {
    Pointer(DType),
    Variable { ty: DType, p2size: u8 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pointwise;
impl TypeSystem for Pointwise {
    type Type = PType;
    type OpData = PointwiseOp;
}

pub struct PointwiseIR {
    ir: IR<Pointwise>,
    size: Size,
    tid: NodeId,
    var: NodeId,
    bufs: Vec<NodeId>,
    read_from: HashSet<NodeId>,
    written_to: HashSet<NodeId>,
}

impl PointwiseIR {
    pub fn new(size: Size) -> Result<Self, IRError> {
        let mut ir = IR::default();

        let [var] = ir.add_op([], PointwiseOp::VarSize)?[..] else {
            return Err("Invalid number outputs!".into());
        };

        let [tid] = ir.add_op([var], PointwiseOp::ThreadId)?[..] else {
            return Err("Invalid number outputs!".into());
        };

        Ok(Self { ir, tid, var, size, bufs: Vec::new(), read_from: HashSet::new(), written_to: HashSet::new() })
    }

    pub fn tid(&self) -> NodeId {
        self.tid
    }

    pub fn var(&self) -> NodeId {
        self.var
    }

    pub fn add_buf(&mut self, ty: TType) -> NodeId {
        let node = self.ir.add_op([], PointwiseOp::Buffer(ty.dtype(), ty.size())).unwrap()[0];
        self.bufs.push(node);
        node
    }

    pub fn add_const(&mut self, value: DValue, p2size: u8) -> NodeId {
        self.ir.add_op([], PointwiseOp::Constant { value, p2size }).unwrap()[0]
    }

    pub fn read(&mut self, buf: NodeId, idx: NodeId, p2size: u8) -> Result<NodeId, IRError> {
        let PType::Pointer(buf_ty) = self.ir.node(buf)?.ty() else {
            return Err("Only buffers allowed as input here!".into());
        };

        let PType::Variable { ty: DType::I32, p2size: 0 } = self.ir.node(idx)?.ty() else {
            return Err("Only scalar integers allowed as indices!".into());
        };

        let io = MemIO { buf_ty, p2size };

        self.read_from.insert(buf);
        self.ir.add_op([buf, idx], PointwiseOp::Read(io)).map(|x| x[0])
    }

    pub fn write(&mut self, buf: NodeId, idx: NodeId, val: NodeId) -> Result<(), IRError> {
        let PType::Pointer(buf_ty) = self.ir.node(buf)?.ty() else {
            return Err("Only buffers allowed as input here!".into());
        };

        let PType::Variable { ty: DType::I32, p2size: 0 } = self.ir.node(idx)?.ty() else {
            return Err("Only scalar integers allowed as indices!".into());
        };

        let PType::Variable { ty: cmp_ty, p2size } = self.ir.node(val)?.ty() else {
            return Err("Only scalar integers allowed as indices!".into());
        };

        if cmp_ty != buf_ty {
            return Err("Mismatched buffer and value types!".into());
        }

        let io = MemIO { buf_ty, p2size };

        self.written_to.insert(buf);
        self.ir.add_op([buf, idx, val], PointwiseOp::Write(io)).map(|_| ())
    }

    pub fn unary(&mut self, node: NodeId, op: Unary) -> Result<NodeId, IRError> {
        let PType::Variable { ty, p2size } = self.ir.node(node)?.ty() else {
            return Err("Only variables allowed in unary ops!".into());
        };

        self.ir.add_op([node], PointwiseOp::Unary { ty, p2size, op }).map(|x| x[0])
    }

    pub fn binary(&mut self, lhs: NodeId, rhs: NodeId, op: CABinary) -> Result<NodeId, IRError> {
        let PType::Variable { ty, p2size } = self.ir.node(lhs)?.ty() else {
            return Err("Only variables allowed in unary ops!".into());
        };

        let PType::Variable { ty: ty2, p2size: p2size2 } = self.ir.node(rhs)?.ty() else {
            return Err("Only variables allowed in unary ops!".into());
        };

        if ty != ty2 || p2size != p2size2 {
            return Err("Mismatched types!".into());
        }

        self.ir.add_op([lhs, rhs], PointwiseOp::Binary { ty, p2size, op }).map(|x| x[0])
    }

    pub fn source_code(&self) -> Result<String, fmt::Error> {
        let name = |id: NodeId| format!("n{}", id.inner());
        let mut code = String::new();

        for op_id in self.ir.topo_order_ops().unwrap() {
            let op = self.ir.op(op_id).unwrap();

            if let PointwiseOp::VarSize | PointwiseOp::Buffer { .. } = op.data() {
            } else {
                println!("{:?}", op.data());
                let mut src = code_str(*op.data(), self.size).unwrap();

                for (i, &id) in op.inputs().iter().enumerate() {
                    src = src.replace(&format!("IN{}", i + 1), &name(id));
                }

                for (i, &id) in op.outputs().iter().enumerate() {
                    src = src.replace(&format!("OUT{}", i + 1), &name(id));
                }

                writeln!(&mut code, "{src}")?;
            }
        }

        let mut src = String::new();

        write!(&mut src, "extern \"C\" __global__ void kernel(")?;

        if self.size.var_power() > 0 {
            write!(&mut src, "const int {}", name(self.var))?;
        }

        for &input in &self.bufs {
            let PType::Pointer(ty) = self.ir.node(input).unwrap().ty() else { panic!() };
            write!(&mut src, ", {}* {}", tystr(ty), name(input))?;
        }

        writeln!(&mut src, ") {{")?;

        for line in code.lines() {
            writeln!(&mut src, "    {line}")?;
        }

        writeln!(&mut src, "}}")?;

        Ok(src)
    }

    pub fn lower(&self) -> Result<KernelSrc, IRError> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut arg_order = Vec::new();

        for &buf in &self.bufs {
            let rf = self.read_from.contains(&buf);
            let wt = self.written_to.contains(&buf);

            if rf && wt {
                return Err("Buffer read from and written to!".into());
            }

            let PointwiseOp::Buffer(dtype, size) = *self.ir.op(self.ir.parent_op(buf)?)?.data() else { panic!() };

            if rf {
                arg_order.push((inputs.len(), true));
                inputs.push(TType::new(size, dtype));
            } else if wt {
                arg_order.push((outputs.len(), false));
                outputs.push(TType::new(size, dtype));
            } else {
                return Err("Unused buffer!".into());
            }
        }

        let source = self.source_code().map_err(|e| IRError::from(format!("{e:?}")))?;

        unsafe {
            Ok(KernelSrc::new(
                inputs,
                outputs,
                source,
                self.size.var_power() > 0,
                arg_order,
                Default::default(),
                Rc::new(|s| Dim3 { x: s.div_ceil(256) as u32, y: 1, z: 1 }),
                Rc::new(|_| 256),
                Rc::new(|_| 0),
            ))
        }
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
        let ty = TType::new(Size::variable(), DType::F32);

        let mut ir = PointwiseIR::new(Size::variable()).unwrap();
        let input1 = ir.add_buf(ty);
        let input2 = ir.add_buf(ty);
        let output = ir.add_buf(ty);

        let tid = ir.tid();

        let scalar = ir.add_const(2.0.into(), 2);
        let value1 = ir.read(input1, tid, 2).unwrap();
        let value2 = ir.read(input2, tid, 2).unwrap();

        let value3 = ir.binary(scalar, value1, CABinary::Mul).unwrap();
        let value4 = ir.binary(value3, value2, CABinary::Add).unwrap();
        ir.write(output, tid, value4).unwrap();

        let source = ir.source_code().unwrap();

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

            let size = 1i32;
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
