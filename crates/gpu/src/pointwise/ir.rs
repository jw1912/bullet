use std::{
    collections::BTreeSet,
    fmt::{self, Write},
    rc::Rc,
};

use bullet_compiler::{
    ir::{IR, IRError, NodeId, TypeSystem},
    tensor::{
        DType, DValue, IRTrace, Size, TType,
        operation::{CABinary, SparseMatmul, Unary},
    },
};

use crate::{
    kernel::KernelSrc,
    pointwise::{
        operations::{MemIO, PType, PointwiseOp},
        write::{code_str, tystr},
    },
    runtime::Dim3,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pointwise;
impl TypeSystem for Pointwise {
    type Type = PType;
    type OpData = PointwiseOp;
}

#[derive(Clone, Debug)]
pub struct PointwiseIR {
    ir: IR<Pointwise>,
    size: Size,
    tid: NodeId,
    var: NodeId,
    bufs: Vec<NodeId>,
    read_from: BTreeSet<NodeId>,
    written_to: BTreeSet<NodeId>,
    needs_zero: BTreeSet<NodeId>,
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

        Ok(Self {
            ir,
            tid,
            var,
            size,
            bufs: Vec::new(),
            read_from: BTreeSet::new(),
            written_to: BTreeSet::new(),
            needs_zero: BTreeSet::new(),
        })
    }

    pub fn tid(&self) -> NodeId {
        self.tid
    }

    pub fn var(&self) -> NodeId {
        self.var
    }

    pub fn eval_size(&mut self, size: Size) -> NodeId {
        self.ir.add_op([self.var], PointwiseOp::EvalSize(size)).unwrap()[0]
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

    pub fn conditional_read(
        &mut self,
        buf: NodeId,
        idx: NodeId,
        cond: NodeId,
        fallback: DValue,
        p2size: u8,
    ) -> Result<NodeId, IRError> {
        let PType::Pointer(buf_ty) = self.ir.node(buf)?.ty() else {
            return Err("Only buffers allowed as input here!".into());
        };

        let PType::Variable { ty: DType::I32, p2size: 0 } = self.ir.node(idx)?.ty() else {
            return Err("Only scalar integers allowed as indices!".into());
        };

        let PType::Variable { ty: DType::I32, p2size: 0 } = self.ir.node(cond)?.ty() else {
            return Err("Only scalar integers allowed as condition!".into());
        };

        if fallback.dtype() != buf_ty {
            return Err("Fallback value does not match dtype!".into());
        }

        let io = MemIO { buf_ty, p2size };

        self.read_from.insert(buf);
        self.ir.add_op([buf, idx, cond], PointwiseOp::ConditionalRead(io, fallback)).map(|x| x[0])
    }

    pub fn write(&mut self, buf: NodeId, idx: NodeId, val: NodeId) -> Result<(), IRError> {
        let PType::Pointer(buf_ty) = self.ir.node(buf)?.ty() else {
            return Err("Only buffers allowed as input here!".into());
        };

        let PType::Variable { ty: DType::I32, p2size: 0 } = self.ir.node(idx)?.ty() else {
            return Err("Only scalar integers allowed as indices!".into());
        };

        let PType::Variable { ty: cmp_ty, p2size } = self.ir.node(val)?.ty() else {
            return Err("Mismatched types!".into());
        };

        if cmp_ty != buf_ty {
            return Err("Mismatched buffer and value types!".into());
        }

        let io = MemIO { buf_ty, p2size };

        self.written_to.insert(buf);
        self.ir.add_op([buf, idx, val], PointwiseOp::Write(io)).map(|_| ())
    }

    pub fn atomic_add(&mut self, buf: NodeId, idx: NodeId, val: NodeId) -> Result<(), IRError> {
        let PType::Pointer(buf_ty) = self.ir.node(buf)?.ty() else {
            return Err("Only buffers allowed as input here!".into());
        };

        let PType::Variable { ty: DType::I32, p2size: 0 } = self.ir.node(idx)?.ty() else {
            return Err("Only scalar integers allowed as indices!".into());
        };

        let PType::Variable { ty: cmp_ty, p2size } = self.ir.node(val)?.ty() else {
            return Err("Mismatched types!".into());
        };

        if cmp_ty != buf_ty {
            return Err("Mismatched buffer and value types!".into());
        }

        let io = MemIO { buf_ty, p2size };

        self.written_to.insert(buf);
        self.needs_zero.insert(buf);
        self.ir.add_op([buf, idx, val], PointwiseOp::AtomicAdd(io)).map(|_| ())
    }

    pub fn broadcast(&mut self, node: NodeId, p2size: u8) -> Result<NodeId, IRError> {
        let PType::Variable { ty, p2size: 0 } = self.ir.node(node)?.ty() else {
            return Err("Only scalar variables allowed in broadcast!".into());
        };

        self.ir.add_op([node], PointwiseOp::Broadcast(ty, p2size.try_into().unwrap())).map(|x| x[0])
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

    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, IRError> {
        let PType::Variable { ty: DType::I32, p2size: 0 } = self.ir.node(lhs)?.ty() else {
            return Err("Only scalar integer variables allowed in div!".into());
        };

        let PType::Variable { ty: DType::I32, p2size: 0 } = self.ir.node(rhs)?.ty() else {
            return Err("Only scalar integer variables allowed in div!".into());
        };

        self.ir.add_op([lhs, rhs], PointwiseOp::Div).map(|x| x[0])
    }

    pub fn rem(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, IRError> {
        let PType::Variable { ty: DType::I32, p2size: 0 } = self.ir.node(lhs)?.ty() else {
            return Err("Only scalar integer variables allowed in rem!".into());
        };

        let PType::Variable { ty: DType::I32, p2size: 0 } = self.ir.node(rhs)?.ty() else {
            return Err("Only scalar integer variables allowed in rem!".into());
        };

        self.ir.add_op([lhs, rhs], PointwiseOp::Rem).map(|x| x[0])
    }

    pub fn sparse_matmul(
        &mut self,
        weights: NodeId,
        indices: NodeId,
        p2size: u8,
        matmul: SparseMatmul,
    ) -> Result<NodeId, IRError> {
        let PType::Pointer(ty) = self.ir.node(weights)?.ty() else {
            return Err("Only pointer allowed!".into());
        };

        let PType::Pointer(DType::I32) = self.ir.node(indices)?.ty() else {
            return Err("Only integer pointer allowed!".into());
        };

        let op = PointwiseOp::SpMM { nnz: matmul.nnz(), rows: matmul.rows(), cols: matmul.cols(), ty, p2size };

        self.read_from.insert(weights);
        self.read_from.insert(indices);
        self.ir.add_op([weights, indices, self.tid], op).map(|x| x[0])
    }

    pub fn sparse_matmul_bwd(
        &mut self,
        weights: NodeId,
        indices: NodeId,
        gradients: NodeId,
        matmul: SparseMatmul,
    ) -> Result<(), IRError> {
        let PType::Pointer(ty) = self.ir.node(weights)?.ty() else {
            return Err("Only pointer allowed!".into());
        };

        let PType::Pointer(DType::I32) = self.ir.node(indices)?.ty() else {
            return Err("Only integer pointer allowed!".into());
        };

        let op = PointwiseOp::SpMMT { nnz: matmul.nnz(), rows: matmul.rows(), cols: matmul.cols(), ty };

        self.needs_zero.insert(weights);
        self.read_from.insert(indices);
        self.written_to.insert(weights);
        self.ir.add_op([weights, indices, self.tid, gradients], op).map(|_| ())
    }

    pub fn eliminate_common_subexprs(&mut self) -> Result<(), IRTrace> {
        while self.eliminate_single_common_subexpr()? {}
        Ok(())
    }

    fn eliminate_single_common_subexpr(&mut self) -> Result<bool, IRTrace> {
        let ops = self.ir.operations().cloned();

        for (i, op_i) in ops.clone().enumerate() {
            for op_j in ops.clone().skip(i + 1) {
                if op_i.inputs() == op_j.inputs() && op_i.data() == op_j.data() && !op_i.data().is_unique() {
                    for (&out_i, &out_j) in op_i.outputs().iter().zip(op_j.outputs()) {
                        self.ir.replace_input_no_cycle_check(out_i, out_j)?;
                    }

                    self.ir.remove_op(op_j.id())?;

                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    pub fn source_code(&self, kernel_name: &str) -> Result<String, fmt::Error> {
        let name = |id: NodeId| format!("n{}", id.inner());
        let mut code = String::new();

        for op_id in self.ir.topo_order_ops().unwrap() {
            let op = self.ir.op(op_id).unwrap();

            if let PointwiseOp::VarSize | PointwiseOp::Buffer { .. } = op.data() {
            } else {
                let mut src = code_str(*op.data(), self.size).unwrap();

                for (i, &id) in op.inputs().iter().enumerate() {
                    src = src.replace(&format!("IN{}", i + 1), &name(id));
                }

                for (i, &id) in op.outputs().iter().enumerate() {
                    src = src.replace(&format!("OUT{}", i + 1), &name(id));
                }

                src = src.replace("UNIQ", &format!("uniq_{}_", op_id.inner()));

                writeln!(&mut code, "{src}")?;
            }
        }

        let mut src = String::new();

        write!(&mut src, "extern \"C\" __global__ void {kernel_name}(")?;

        let varp = self.size.var_power();
        if varp > 0 {
            write!(&mut src, "const int {}", name(self.var))?;
        }

        for (i, &input) in self.bufs.iter().enumerate() {
            let comma = if i > 0 || varp > 0 { ", " } else { "" };
            let PType::Pointer(ty) = self.ir.node(input).unwrap().ty() else { panic!() };
            write!(&mut src, "{comma}{}* {}", tystr(ty), name(input))?;
        }

        writeln!(&mut src, ") {{")?;

        for line in code.lines() {
            writeln!(&mut src, "    {line}")?;
        }

        writeln!(&mut src, "}}")?;

        Ok(src)
    }

    pub fn estimate_memory_cost(&self) -> Result<Size, IRError> {
        let mut cost = 0;

        let estcost = |bytes, p2size| bytes * [8, 7, 6][usize::from(p2size)];

        for op in self.ir.operations() {
            use PointwiseOp as P;
            match *op.data() {
                P::Read(io) | P::Write(io) | P::AtomicAdd(io) | P::ConditionalRead(io, _) => {
                    let bytes_per_thread = io.buf_ty.bytes() * 2usize.pow(u32::from(io.p2size));
                    cost += estcost(bytes_per_thread, io.p2size);
                }
                P::SpMM { nnz, ty, p2size, .. } => {
                    let bytes_per_thread = nnz * ty.bytes() * 2usize.pow(u32::from(p2size));
                    cost += estcost(bytes_per_thread, p2size);
                }
                P::SpMMT { nnz, ty, .. } => {
                    cost += estcost(nnz * ty.bytes(), 0);
                }
                P::Unary { .. }
                | P::Binary { .. }
                | P::ThreadId
                | P::Constant { .. }
                | P::Buffer { .. }
                | P::VarSize
                | P::Div
                | P::Rem
                | P::EvalSize(_)
                | P::Broadcast(_, _) => {}
            }
        }

        Ok(self.size * cost)
    }

    /// ### Safety
    ///
    /// User must ensure same invariants as KernelSrc hold
    pub unsafe fn lower(&self, name: String) -> Result<KernelSrc, IRError> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut arg_order = Vec::new();
        let mut requires_zero = BTreeSet::new();

        for &buf in &self.bufs {
            let rf = self.read_from.contains(&buf);
            let wt = self.written_to.contains(&buf);

            let PointwiseOp::Buffer(dtype, size) = *self.ir.op(self.ir.parent_op(buf)?)?.data() else { panic!() };

            if wt {
                arg_order.push((outputs.len(), false));

                if self.needs_zero.contains(&buf) {
                    requires_zero.insert(outputs.len());
                }

                outputs.push(TType::new(size, dtype));
            } else if rf {
                arg_order.push((inputs.len(), true));
                inputs.push(TType::new(size, dtype));
            } else {
                return Err("Unused buffer!".into());
            }
        }

        let source = self.source_code(&name).map_err(|e| IRError::from(format!("{e:?}")))?;
        let total = self.size;

        unsafe {
            Ok(KernelSrc::new(
                inputs,
                outputs,
                name,
                source,
                self.size.var_power() > 0,
                arg_order,
                requires_zero,
                Rc::new(move |s| Dim3 { x: total.evaluate(s).div_ceil(256) as u32, y: 1, z: 1 }),
                Rc::new(|_| 256),
                Rc::new(|_| 0),
            ))
        }
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[cfg(test)]
mod tests {
    use bullet_compiler::tensor::TValue;

    use crate::{
        buffer::Buffer,
        runtime::{Device, Gpu, Stream},
    };

    use super::*;

    fn fmadd<G: Gpu>() -> Result<(), G::Error> {
        let ty = TType::new(Size::variable(), DType::F32);

        let mut ir = PointwiseIR::new(Size::variable()).unwrap();
        let input1 = ir.add_buf(ty);
        let input2 = ir.add_buf(ty);

        let tid = ir.tid();

        let scalar = ir.add_const(2.0.into(), 2);
        let value1 = ir.read(input1, tid, 2).unwrap();
        let value2 = ir.read(input2, tid, 2).unwrap();

        let value3 = ir.binary(scalar, value1, CABinary::Mul).unwrap();
        let value4 = ir.binary(value3, value2, CABinary::Add).unwrap();
        ir.write(input2, tid, value4).unwrap();

        let device = Device::<G>::new(0)?;
        let kernel = unsafe { ir.lower("fmadd".to_string()).unwrap() }.compile(device.clone())?;
        let stream = Stream::new(device)?;

        let values1 = TValue::F32(vec![1.0, 2.0, 3.0, 4.0]);
        let values2 = TValue::F32(vec![4.0, 3.0, 2.0, 1.0]);

        let input1_buf = Buffer::from_host(&stream, &values1)?.value()?.0;
        let input2_buf = Buffer::from_host(&stream, &values2)?.value()?.0;

        kernel.execute(stream.clone(), vec![input1_buf], vec![input2_buf.clone()])?.value()?;

        let actual = input2_buf.to_host(&stream)?.value()?;

        assert_eq!(actual, TValue::F32(vec![6.0, 7.0, 8.0, 9.0]));

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
