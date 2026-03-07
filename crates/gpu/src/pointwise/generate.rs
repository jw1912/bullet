use std::collections::{HashMap, HashSet};

use bullet_compiler::tensor::{
    DValue, IRTrace, Size,
    operation::{
        BroadcastAcrossDimension, CABinary, CABinaryOp, PadAcrossDimension, ScalarConstant, SparseMatmul,
        SparseMatmulBwd, SubGraph, Unary, UnaryOp,
    },
};

use super::PointwiseIR;

pub fn generate(sub: &SubGraph) -> Result<Option<PointwiseIR>, IRTrace> {
    let ir = sub.internal_graph();

    let mut size = None;
    let mut p2size = 2;

    for op in ir.ordered_operations()? {
        let data = op.data();

        let (new_size, align) = if op.data().is_input() {
            continue;
        } else if let Some(scalar) = data.downcast::<ScalarConstant>() {
            (scalar.1, scalar.1.factor())
        } else if let Some(broadcast) = data.downcast::<BroadcastAcrossDimension>() {
            if !ir.is_input(op.inputs()[0])? {
                return Ok(None);
            }

            let factor = if broadcast.inner() == Size::constant(1) {
                broadcast.repeats().factor()
            } else {
                broadcast.inner().factor()
            };

            (broadcast.output_size(), factor)
        } else if let Some(binary) = data.downcast::<CABinaryOp>() {
            let size = binary.ty().size();
            (size, size.factor())
        } else if let Some(unary) = data.downcast::<UnaryOp>() {
            let size = unary.output_type().size();
            (size, size.factor())
        } else if let Some(matmul) = data.downcast::<SparseMatmul>() {
            if !(ir.is_input(op.inputs()[0])? && ir.is_input(op.inputs()[1])?) {
                return Ok(None);
            }

            (matmul.batch() * matmul.rows(), matmul.rows())
        } else if let Some(bwd) = data.downcast::<SparseMatmulBwd>() {
            if !(ir.is_input(op.inputs()[1])? && ir.is_output(op.outputs()[0])) {
                return Ok(None);
            }

            (bwd.0.batch() * bwd.0.rows(), 1)
        } else if let Some(pad) = data.downcast::<PadAcrossDimension>() {
            if !ir.is_input(op.inputs()[0])? {
                return Ok(None);
            }

            (pad.output_size(), 1)
        } else {
            return Ok(None);
        };

        p2size = p2size.min(align.trailing_zeros());
        if let Some(size) = size {
            if size != new_size {
                return Ok(None);
            }
        } else {
            size = Some(new_size);
        }
    }

    let Some(size) = size else { return Ok(None) };
    let p2actual = 2usize.pow(p2size);
    let p2size = p2size as u8;

    let mut pntwise = PointwiseIR::new(size / p2actual)?;
    let mut mapping = HashMap::new();

    let inp_buf_map: HashMap<_, _> =
        sub.internal_inputs().iter().map(|&i| (i, pntwise.add_buf(ir.get_node(i).unwrap().ty()))).collect();

    let out_buf_map: HashMap<_, _> =
        sub.internal_outputs().iter().map(|&o| (o, pntwise.add_buf(ir.get_node(o).unwrap().ty()))).collect();

    let get_val = |node, pntw: &mut PointwiseIR, map: &HashMap<_, _>| {
        if ir.is_input(node)? {
            let buf = *inp_buf_map.get(&node).unwrap();
            pntw.read(buf, pntw.tid(), p2size).map(Option::Some).map_err(IRTrace::from)
        } else {
            Ok(map.get(&node).cloned())
        }
    };

    let mut handled_writes = HashSet::new();

    for op in ir.ordered_operations()? {
        let data = op.data();

        if op.data().is_input() {
        } else if let Some(scalar) = data.downcast::<ScalarConstant>() {
            let scalar = pntwise.add_const(scalar.0, p2size);
            mapping.insert(op.outputs()[0], scalar);
        } else if let Some(broadcast) = data.downcast::<BroadcastAcrossDimension>() {
            let out = op.outputs()[0];
            let buf = *inp_buf_map.get(&op.inputs()[0]).unwrap();

            let tid = pntwise.tid();

            // special case, if we are reding an inner scalar and repeating a
            // multiple of 2^N times, we can read the scalar and broadcast it
            // into the appropriate p2 size to avoid killing the vectorization
            // on the rest of the kernel
            if p2size > 0 && broadcast.inner() == Size::constant(1) {
                let repeats = pntwise.eval_size(broadcast.repeats() / p2actual);
                let idx = pntwise.div(tid, repeats)?;
                let scalar = pntwise.read(buf, idx, 0)?;
                let output = pntwise.broadcast(scalar, p2size)?;
                mapping.insert(out, output);
            } else {
                let repeats = pntwise.eval_size(broadcast.repeats());
                let inner = pntwise.eval_size(broadcast.inner() / p2actual);
                let oidx_denom = pntwise.binary(repeats, inner, CABinary::Mul)?;
                let oidx = pntwise.div(tid, oidx_denom)?;
                let iidx = pntwise.rem(tid, inner)?;
                let idx_base = pntwise.binary(inner, oidx, CABinary::Mul)?;
                let idx = pntwise.binary(idx_base, iidx, CABinary::Add)?;
                let output = pntwise.read(buf, idx, p2size)?;
                mapping.insert(out, output);
            }
        } else if let Some(binary) = data.downcast::<CABinaryOp>() {
            let out = op.outputs()[0];
            let Some(lhs) = get_val(op.inputs()[0], &mut pntwise, &mapping)? else { return Ok(None) };
            let Some(rhs) = get_val(op.inputs()[1], &mut pntwise, &mapping)? else { return Ok(None) };

            let output = pntwise.binary(lhs, rhs, binary.op())?;
            mapping.insert(out, output);
        } else if let Some(unary) = data.downcast::<UnaryOp>() {
            let out = op.outputs()[0];
            let Some(input) = get_val(op.inputs()[0], &mut pntwise, &mapping)? else { return Ok(None) };

            let output = pntwise.unary(input, unary.op())?;
            mapping.insert(out, output);
        } else if let Some(matmul) = data.downcast::<SparseMatmul>() {
            let weights = *inp_buf_map.get(&op.inputs()[0]).unwrap();
            let indices = *inp_buf_map.get(&op.inputs()[1]).unwrap();
            let output = pntwise.sparse_matmul(weights, indices, p2size, *matmul)?;
            mapping.insert(op.outputs()[0], output);
        } else if let Some(bwd) = data.downcast::<SparseMatmulBwd>() {
            let out = op.outputs()[0];
            let out_id = *out_buf_map.get(&out).unwrap();

            let weights = *out_buf_map.get(&out).unwrap();
            let indices = *inp_buf_map.get(&op.inputs()[1]).unwrap();
            let Some(gradients) = get_val(op.inputs()[0], &mut pntwise, &mapping)? else { return Ok(None) };

            pntwise.sparse_matmul_bwd(weights, indices, gradients, bwd.0)?;
            handled_writes.insert(out_id);
        } else if let Some(pad) = data.downcast::<PadAcrossDimension>() {
            assert_eq!(p2size, 0);
            let buf = *inp_buf_map.get(&op.inputs()[0]).unwrap();

            let before = i32::try_from(pad.before).unwrap();
            let after = i32::try_from(pad.after).unwrap();
            let dimen = i32::try_from(pad.dimen).unwrap();
            let inner = pntwise.eval_size(pad.inner);

            let bda = pntwise.add_const(DValue::I32(before + dimen + after), 0);
            let stride = pntwise.binary(inner, bda, CABinary::Mul)?;

            let tid = pntwise.tid();
            let idx_outer = pntwise.div(tid, stride)?;
            let idx_non_outer = pntwise.rem(tid, stride)?;
            let idx_bda = pntwise.div(idx_non_outer, inner)?;
            let idx_inner = pntwise.rem(idx_non_outer, inner)?;

            let offset = pntwise.add_const(DValue::I32(-before), 0);
            let idx_dimen = pntwise.binary(idx_bda, offset, CABinary::Add)?;

            let dimen = pntwise.add_const(DValue::I32(dimen), 0);
            let mut idx = pntwise.binary(dimen, idx_outer, CABinary::Mul)?;
            idx = pntwise.binary(idx, idx_dimen, CABinary::Add)?;
            idx = pntwise.binary(idx, inner, CABinary::Mul)?;
            idx = pntwise.binary(idx, idx_inner, CABinary::Add)?;

            let p1 = pntwise.unary(idx_dimen, Unary::IsNonNegative)?;

            let minus_one = pntwise.add_const(DValue::I32(-1), 0);
            let neg_idx = pntwise.binary(minus_one, idx_dimen, CABinary::Mul)?;
            let overflow = pntwise.binary(dimen, neg_idx, CABinary::Add)?;
            let p2 = pntwise.unary(overflow, Unary::IsPositive)?;

            let cond = pntwise.binary(p1, p2, CABinary::Mul)?;

            let output = pntwise.conditional_read(buf, idx, cond, pad.value, p2size)?;
            mapping.insert(op.outputs()[0], output);
        } else {
            unreachable!();
        };

        for &output in op.outputs() {
            if ir.is_output(output) && !handled_writes.contains(&output) {
                let id = *out_buf_map.get(&output).unwrap();
                let Some(val) = get_val(output, &mut pntwise, &mapping)? else { return Ok(None) };
                pntwise.write(id, pntwise.tid(), val)?;
            }
        }
    }

    pntwise.eliminate_common_subexprs()?;

    Ok(Some(pntwise))
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[cfg(test)]
mod tests {
    use bullet_compiler::tensor::{DType, DValue, IRBuilder, IRTrace, Size, TValue, operation::SubGraph};

    use crate::{
        buffer::Buffer,
        kernel::KernelSrc,
        runtime::{Device, Gpu, Stream},
    };

    fn make_axby() -> Result<KernelSrc, IRTrace> {
        let size = Size::variable() * 4;
        let builder = IRBuilder::default();

        let a = builder.add_input(8, DType::F32);
        let b = builder.add_input(8, DType::F32);
        let x = builder.add_input(size * 8, DType::F32);
        let y = ((a.broadcast([8], 0, size)? * x)? + b.broadcast([8, 1], 1, size)?)?;
        let ir = builder.build([x, y]);

        let sub = SubGraph::new(ir, vec![a.node(), b.node(), x.node()], vec![x.node(), y.node()])?;
        unsafe { super::generate(&sub)?.unwrap().lower().map_err(IRTrace::from) }
    }

    fn axby<G: Gpu>() -> Result<(), G::Error> {
        let device = Device::<G>::new(0)?;
        let stream = Stream::new(device.clone())?;

        let src = make_axby().unwrap();

        let axby = src.compile(device)?;

        let aval = TValue::F32(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        let bval = TValue::F32(vec![4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0]);
        let xval = TValue::F32([4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0].repeat(4));

        let aval_buf = Buffer::from_host(&stream, &aval)?.value().0;
        let bval_buf = Buffer::from_host(&stream, &bval)?.value().0;
        let xval_buf = Buffer::from_host(&stream, &xval)?.value().0;

        let x2val_buf = Buffer::zeroed(&stream, xval.dtype(), xval.size())?.value();
        let yval_buf = Buffer::zeroed(&stream, xval.dtype(), xval.size())?.value();

        let sync = axby.execute(
            stream.clone(),
            vec![aval_buf, bval_buf, xval_buf],
            vec![x2val_buf.clone(), yval_buf.clone()],
        )?;

        drop(sync);

        let actualx = x2val_buf.to_host(&stream)?.value();
        let actualy = yval_buf.to_host(&stream)?.value();

        assert_eq!(actualx, xval);
        #[rustfmt::skip]
        assert_eq!(actualy, TValue::F32(vec![
            8.0, 10.0, 10.0, 8.0, 7.0, 9.0, 9.0, 7.0,
            6.0, 8.0, 8.0, 6.0, 5.0, 7.0, 7.0, 5.0,
            8.0, 10.0, 10.0, 8.0, 7.0, 9.0, 9.0, 7.0,
            6.0, 8.0, 8.0, 6.0, 5.0, 7.0, 7.0, 5.0,
        ]));

        Ok(())
    }

    fn make_concat(dim: usize) -> Result<KernelSrc, IRTrace> {
        let builder = IRBuilder::default();

        let a = builder.add_input(8, DType::F32);
        let b = builder.add_input(8, DType::F32);

        let apad = a.pad([2, 2, 2], dim, 0, 2, DValue::F32(0.0))?;
        let bpad = b.pad([2, 2, 2], dim, 2, 0, DValue::F32(0.0))?;
        let concat = (apad + bpad)?;

        let ir = builder.build([concat]);

        let sub = SubGraph::new(ir, vec![a.node(), b.node()], vec![concat.node()])?;
        unsafe { super::generate(&sub)?.unwrap().lower().map_err(IRTrace::from) }
    }

    fn concat<G: Gpu>(dim: usize, expected: impl Into<Vec<f32>>) -> Result<(), G::Error> {
        let device = Device::<G>::new(0)?;
        let stream = Stream::new(device.clone())?;

        let src = make_concat(dim).unwrap();

        let concat = src.compile(device)?;

        let aval = TValue::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let bval = TValue::F32(vec![4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0]);

        let aval_buf = Buffer::from_host(&stream, &aval)?.value().0;
        let bval_buf = Buffer::from_host(&stream, &bval)?.value().0;

        let concat_buf = Buffer::zeroed(&stream, DType::F32, 16)?.value();

        let sync = concat.execute(stream.clone(), vec![aval_buf, bval_buf], vec![concat_buf.clone()])?;

        drop(sync);

        assert_eq!(concat_buf.to_host(&stream)?.value(), TValue::F32(expected.into()));

        Ok(())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::runtime::cuda::{Cuda, CudaError};

        #[test]
        fn axby() -> Result<(), CudaError> {
            super::axby::<Cuda>()
        }

        #[test]
        fn concat() -> Result<(), CudaError> {
            super::concat::<Cuda>(0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0])?;
            super::concat::<Cuda>(1, [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0])?;
            super::concat::<Cuda>(2, [1.0, 2.0, 4.0, 3.0, 3.0, 4.0, 2.0, 1.0, 5.0, 6.0, 1.0, 2.0, 7.0, 8.0, 3.0, 4.0])
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use crate::runtime::rocm::{ROCm, ROCmError};

        #[test]
        fn axby() -> Result<(), ROCmError> {
            super::axby::<ROCm>()
        }

        #[test]
        fn concat() -> Result<(), ROCmError> {
            super::concat::<ROCm>()
        }
    }
}
