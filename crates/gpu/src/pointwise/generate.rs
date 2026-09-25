use std::collections::{BTreeMap, BTreeSet};

use bullet_compiler::tensor::{
    DValue, IRTrace, Size,
    operation::{
        BroadcastAcrossDimension, CABinaryOp, PadAcrossDimension, Power, ReduceAcrossDimension, Reduction,
        ScalarConstant, Select, SelectPad, SliceAcrossDimension, SparseMatmul, SparseMatmulBwdMulti, SubGraph, UnaryOp,
    },
};

use crate::runtime::DeviceProps;

use super::{PointwiseBuf, PointwiseBuilder, PointwiseIR};

pub fn generate(sub: &SubGraph, props: &DeviceProps) -> Result<Option<(PointwiseIR, bool)>, IRTrace> {
    let ir = sub.internal_graph();

    let mut size = None;
    let mut p2size = 2;

    for op in ir.ordered_operations()? {
        let data = op.data();

        let (new_size, align) = if op.data().is_input() {
            continue;
        } else if let Some(scalar) = data.downcast::<ScalarConstant>() {
            (scalar.1, scalar.1)
        } else if let Some(broadcast) = data.downcast::<BroadcastAcrossDimension>() {
            if !ir.is_input(op.inputs()[0])? {
                return Ok(None);
            }

            let factor = if broadcast.inner().get() == 1 { broadcast.repeats() } else { broadcast.inner() };

            (broadcast.output_size(), factor)
        } else if let Some(binary) = data.downcast::<CABinaryOp>() {
            let size = binary.ty().size();
            (size, size)
        } else if let Some(&Power(size)) = data.downcast::<Power>() {
            (size, size)
        } else if let Some(unary) = data.downcast::<UnaryOp>() {
            let size = unary.output_type().size();
            (size, size)
        } else if let Some(matmul) = data.downcast::<SparseMatmul>() {
            if !(ir.is_input(op.inputs()[0])? && ir.is_input(op.inputs()[1])?) {
                return Ok(None);
            }

            let hp2 = matmul
                .rows()
                .get()
                .trailing_zeros()
                .min(matmul.stride().get().trailing_zeros())
                .min(matmul.offset().trailing_zeros());

            (matmul.batch() * matmul.rows(), (1 << hp2).into())
        } else if let Some(bwd) = data.downcast::<SparseMatmulBwdMulti>() {
            if !ir.is_output(op.outputs()[0]) {
                return Ok(None);
            }

            for &input in op.inputs().iter().skip(1).step_by(2) {
                if !ir.is_input(input)? {
                    return Ok(None);
                }
            }

            (bwd.batch() * bwd.rows(), Size::from(1))
        } else if let Some(pad) = data.downcast::<PadAcrossDimension>() {
            if !ir.is_input(op.inputs()[0])? {
                return Ok(None);
            }

            (pad.output_size(), Size::from(1))
        } else if let Some(slice) = data.downcast::<SliceAcrossDimension>() {
            if !ir.is_input(op.inputs()[0])? {
                return Ok(None);
            }

            (slice.output_size(), Size::from(1))
        } else if let Some(select) = data.downcast::<Select>() {
            if !(ir.is_input(op.inputs()[0])? && ir.is_input(op.inputs()[1])?) {
                return Ok(None);
            }

            (select.output_size(), Size::from(1))
        } else if let Some(select_pad) = data.downcast::<SelectPad>() {
            if !(ir.is_input(op.inputs()[0])? && ir.is_input(op.inputs()[1])?) {
                return Ok(None);
            }

            (select_pad.output_size(), Size::from(1))
        } else if let Some(reduce) = data.downcast::<ReduceAcrossDimension>() {
            if let Some(warp_size) = props.warp_size()
                && reduce.reduction() == Reduction::Sum
                && ir.is_output(op.outputs()[0])
                && reduce.inner().is_multiple_of(usize::from(warp_size).into())
            {
                (reduce.input_size(), Size::from(1))
            } else {
                return Ok(None);
            }
        } else {
            return Ok(None);
        };

        p2size = p2size.min(align.get().trailing_zeros());
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

    let builder = PointwiseBuilder::new(size / p2actual.into());

    let inp_buf_map: BTreeMap<_, _> =
        sub.internal_inputs().iter().map(|&i| (i, builder.new_buffer(ir.get_node(i).unwrap().ty()))).collect();

    let out_buf_map: BTreeMap<_, _> =
        sub.internal_outputs().iter().map(|&o| (o, builder.new_buffer(ir.get_node(o).unwrap().ty()))).collect();

    let get_val = |node, map: &BTreeMap<_, _>| {
        if ir.is_input(node)? {
            let buf: PointwiseBuf = *inp_buf_map.get(&node).unwrap();
            Ok(Some(buf.read(builder.tid(), p2size)))
        } else {
            Ok::<_, IRTrace>(map.get(&node).copied())
        }
    };

    let mut mapping = BTreeMap::new();
    let mut handled_writes = BTreeSet::new();

    for op in ir.ordered_operations()? {
        let data = op.data();

        if op.data().is_input() {
        } else if let Some(scalar) = data.downcast::<ScalarConstant>() {
            let scalar = builder.new_constant(scalar.0, p2size);
            mapping.insert(op.outputs()[0], scalar);
        } else if let Some(broadcast) = data.downcast::<BroadcastAcrossDimension>() {
            let out = op.outputs()[0];
            let buf = *inp_buf_map.get(&op.inputs()[0]).unwrap();

            let tid = builder.tid();

            // special case, if we are reding an inner scalar and repeating a
            // multiple of 2^N times, we can read the scalar and broadcast it
            // into the appropriate p2 size to avoid killing the vectorization
            // on the rest of the kernel
            if p2size > 0 && broadcast.inner().get() == 1 {
                let repeats = i32::try_from(broadcast.repeats().get() / p2actual).unwrap();
                let scalar = buf.read(tid.div(repeats), 0);
                mapping.insert(out, scalar.broadcast(p2size));
            } else {
                let repeats = i32::try_from(broadcast.repeats().get()).unwrap();
                let inner = i32::try_from(broadcast.inner().get() / p2actual).unwrap();

                let oidx = tid.div(repeats * inner);
                let iidx = tid.rem(inner);
                let idx = inner * oidx + iidx;
                mapping.insert(out, buf.read(idx, p2size));
            }
        } else if let Some(binary) = data.downcast::<CABinaryOp>() {
            let out = op.outputs()[0];
            let Some(lhs) = get_val(op.inputs()[0], &mapping)? else { return Ok(None) };
            let Some(rhs) = get_val(op.inputs()[1], &mapping)? else { return Ok(None) };

            mapping.insert(out, lhs.binary(rhs, binary.op()));
        } else if data.downcast::<Power>().is_some() {
            let out = op.outputs()[0];
            let Some(lhs) = get_val(op.inputs()[0], &mapping)? else { return Ok(None) };
            let Some(rhs) = get_val(op.inputs()[1], &mapping)? else { return Ok(None) };

            mapping.insert(out, lhs.powf(rhs));
        } else if let Some(unary) = data.downcast::<UnaryOp>() {
            let out = op.outputs()[0];
            let Some(input) = get_val(op.inputs()[0], &mapping)? else { return Ok(None) };

            mapping.insert(out, input.unary(unary.op()));
        } else if let Some(matmul) = data.downcast::<SparseMatmul>() {
            let weights = *inp_buf_map.get(&op.inputs()[0]).unwrap();
            let indices = *inp_buf_map.get(&op.inputs()[1]).unwrap();
            mapping.insert(op.outputs()[0], weights.sparse_matmul(indices, *matmul, p2size));
        } else if let Some(bwd) = data.downcast::<SparseMatmulBwdMulti>() {
            let out = op.outputs()[0];
            let weights = *out_buf_map.get(&out).unwrap();

            for (this_bwd, inputs) in bwd.inner().iter().zip(op.inputs().chunks_exact(2)) {
                let indices = *inp_buf_map.get(&inputs[1]).unwrap();
                let Some(gradients) = get_val(inputs[0], &mapping)? else { return Ok(None) };
                weights.sparse_matmul_bwd(indices, gradients, this_bwd.0);
            }

            handled_writes.insert(out);
        } else if let Some(pad) = data.downcast::<PadAcrossDimension>() {
            assert_eq!(p2size, 0);
            let buf = *inp_buf_map.get(&op.inputs()[0]).unwrap();

            let before = i32::try_from(pad.before()).unwrap();
            let after = i32::try_from(pad.after()).unwrap();
            let dimen = i32::try_from(pad.dimen().get()).unwrap();
            let inner = i32::try_from(pad.inner().get()).unwrap();

            let (idx_outer, idx_non_outer) = builder.tid().div_rem(inner * (before + dimen + after));
            let (idx_bda, idx_inner) = idx_non_outer.div_rem(inner);

            let idx_dimen = idx_bda - before;
            let idx = (dimen * idx_outer + idx_dimen) * inner + idx_inner;

            // in bounds iff `0 <= idx_dimen < dimen`
            let cond = idx_dimen.is_non_negative() * (dimen - idx_dimen).is_positive();

            mapping.insert(op.outputs()[0], buf.conditional_read(idx, cond, pad.value(), p2size));
        } else if let Some(slice) = data.downcast::<SliceAcrossDimension>() {
            assert_eq!(p2size, 0);
            let buf = *inp_buf_map.get(&op.inputs()[0]).unwrap();

            let inner = i32::try_from(slice.inner().get()).unwrap();
            let slicelen = i32::try_from(slice.end() - slice.start()).unwrap();
            let dimen = i32::try_from(slice.dimen().get()).unwrap();
            let start = i32::try_from(slice.start()).unwrap();

            let (idx_outer, idx_non_outer) = builder.tid().div_rem(inner * slicelen);
            let (idx_slice, idx_inner) = idx_non_outer.div_rem(inner);

            let idx = (dimen * idx_outer + start + idx_slice) * inner + idx_inner;

            mapping.insert(op.outputs()[0], buf.read(idx, p2size));
        } else if let Some(select) = data.downcast::<Select>() {
            assert_eq!(p2size, 0);
            let values = *inp_buf_map.get(&op.inputs()[0]).unwrap();
            let indices = *inp_buf_map.get(&op.inputs()[1]).unwrap();

            let sub_size = i32::try_from((select.inner / select.divisor).get()).unwrap();
            let inner = i32::try_from(select.inner.get()).unwrap();

            let (batch_idx, elem_idx) = builder.tid().div_rem(sub_size);
            let bucket = indices.read(batch_idx, p2size);

            let idx = batch_idx * inner + bucket * sub_size + elem_idx;

            mapping.insert(op.outputs()[0], values.read(idx, p2size));
        } else if let Some(select_pad) = data.downcast::<SelectPad>() {
            assert_eq!(p2size, 0);
            let values = *inp_buf_map.get(&op.inputs()[0]).unwrap();
            let indices = *inp_buf_map.get(&op.inputs()[1]).unwrap();

            let sub_size = i32::try_from((select_pad.inner / select_pad.divisor).get()).unwrap();
            let inner = i32::try_from(select_pad.inner.get()).unwrap();

            let (batch_idx, inner_idx) = builder.tid().div_rem(inner);
            let (bucket_idx, elem_idx) = inner_idx.div_rem(sub_size);

            let bucket_target = indices.read(batch_idx, p2size);
            let cond = (bucket_idx - bucket_target).is_zero();

            let idx = batch_idx * sub_size + elem_idx;
            let fallback = DValue::zero(select_pad.dtype);

            mapping.insert(op.outputs()[0], values.conditional_read(idx, cond, fallback, p2size));
        } else if let Some(reduce) = data.downcast::<ReduceAcrossDimension>() {
            let out = op.outputs()[0];
            let Some(value) = get_val(op.inputs()[0], &mapping)? else { return Ok(None) };
            let dest = *out_buf_map.get(&out).unwrap();

            let dimen = i32::try_from(reduce.dimen().get()).unwrap();
            let inner = i32::try_from(reduce.inner().get()).unwrap();

            let tid = builder.tid();
            let inner_idx = tid.rem(inner);
            let outer_idx = tid.div(inner * dimen);

            dest.atomic_add(inner * outer_idx + inner_idx, value);

            handled_writes.insert(out);
        } else {
            unreachable!();
        };

        for &output in op.outputs() {
            if ir.is_output(output) && !handled_writes.contains(&output) {
                let buf = *out_buf_map.get(&output).unwrap();
                let Some(val) = get_val(output, &mapping)? else { return Ok(None) };
                buf.write(builder.tid(), val);
            }
        }
    }

    builder.ir().eliminate_common_subexprs()?;

    Ok(Some((builder.inner(), p2size > 0)))
}

#[cfg(test)]
mod tests {
    use bullet_compiler::tensor::{
        DType, IRBuilder,
        operation::{SparseMatmul, SubGraph},
    };

    use crate::runtime::{DeviceProps, Dialect};

    // only the tests that compile and run a kernel need a device
    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    use crate::{
        buffer::Buffer,
        kernel::KernelSrc,
        runtime::{Device, Gpu},
    };
    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    use bullet_compiler::tensor::{DValue, IRTrace, TValue};

    fn props() -> DeviceProps {
        DeviceProps::testing(Dialect::CudaHip, Some(32), false)
    }

    /// 32 rows over a vector width of 4 leaves 8 rows per thread.
    fn sparse_matmul_subgraph() -> SubGraph {
        let matmul = SparseMatmul::new(DType::F32, 4usize, 32usize, 8usize, 32usize, 0, 2usize).unwrap();

        let b = IRBuilder::default();
        let w = b.add_input(32 * 8, DType::F32);
        let i = b.add_input(4 * 2, DType::I32);
        let out = b.add_op([w, i], matmul).unwrap()[0];

        SubGraph::new(b.build([out]), vec![w.node(), i.node()], vec![out.node()]).unwrap()
    }

    /// Generated source must not depend on how many other graphs have been built, or a
    /// kernel cannot be meaningfully compared against one from a previous run.
    #[test]
    fn generation_is_deterministic() {
        // the sparse matmul names temporaries after their op, so this covers op ids
        // as well as node ids
        let render = || {
            let ir = super::generate(&sparse_matmul_subgraph(), &props()).unwrap().unwrap().0;
            ir.source_code("kernel", &props()).unwrap()
        };

        let first = render();

        // allocate unrelated nodes and ops in between, bumping any global counters
        for _ in 0..7 {
            let b = IRBuilder::default();
            let x = b.add_input(8, DType::F32);
            let _ = (x * x).unwrap();
        }

        assert_eq!(first, render(), "kernel source depends on unrelated graph construction");
    }

    /// The vectorisation analysis decides how many elements each thread handles, and
    /// silently falling back to scalar is a performance loss rather than a failure, so
    /// it would otherwise go unnoticed.
    #[test]
    fn vector_widths() {
        for (size, vectorised) in [(4usize, true), (8, true), (6, true), (5, false)] {
            let b = IRBuilder::default();
            let x = b.add_input(size, DType::F32);
            let y = (x * x).unwrap();
            let sub = SubGraph::new(b.build([y]), vec![x.node()], vec![y.node()]).unwrap();

            let (_, actual) = super::generate(&sub, &props()).unwrap().unwrap();
            assert_eq!(actual, vectorised, "size {size}");
        }
    }

    /// A reduction only fuses when its inner dimension is wave aligned, so the same
    /// graph fuses on a wave32 device and not on a wave64 one.
    #[test]
    fn reduction_fuses_only_when_wave_aligned() {
        let fuses = |inner: usize, warp_size: Option<u8>| {
            let b = IRBuilder::default();
            let x = b.add_input(inner * 4, DType::F32);
            let r = x.reduce_sum([4, inner], 0).unwrap();
            let sub = SubGraph::new(b.build([r]), vec![x.node()], vec![r.node()]).unwrap();

            let props = DeviceProps::testing(Dialect::CudaHip, warp_size, false);
            super::generate(&sub, &props).unwrap().is_some()
        };

        assert!(fuses(64, Some(32)));
        assert!(fuses(64, Some(64)));
        assert!(fuses(32, Some(32)));
        assert!(!fuses(32, Some(64)));
        assert!(!fuses(12, Some(32)));
        assert!(!fuses(64, None));
    }

    /// The AMD scalar-load hint is only valid on ROCm, and only when the rows each
    /// thread covers make the batch index uniform across the wave.
    #[test]
    fn amd_scalar_load_hint() {
        let emitted = |rows: usize, warp_size: Option<u8>, is_rocm: bool| {
            let matmul = SparseMatmul::new(DType::F32, 4usize, rows, 8usize, rows, 0, 2usize).unwrap();

            let b = IRBuilder::default();
            let w = b.add_input(rows * 8, DType::F32);
            let i = b.add_input(4 * 2, DType::I32);
            let out = b.add_op([w, i], matmul).unwrap()[0];
            let sub = SubGraph::new(b.build([out]), vec![w.node(), i.node()], vec![out.node()]).unwrap();

            let props = DeviceProps::testing(Dialect::CudaHip, warp_size, is_rocm);
            let ir = super::generate(&sub, &props).unwrap().unwrap().0;
            ir.source_code("kernel", &props).unwrap().contains("__builtin_amdgcn_readfirstlane")
        };

        // 256 rows over a vector width of 4 leaves 64 per thread, a multiple of both
        // wave sizes; 32 rows leave 8, a multiple of neither
        assert!(emitted(256, Some(32), true));
        assert!(emitted(256, Some(64), true));
        assert!(!emitted(256, Some(32), false), "not a ROCm builtin");
        assert!(!emitted(256, None, true), "wave size unknown");
        assert!(!emitted(32, Some(32), true), "index not uniform across the wave");
    }

    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    fn make_axby(props: &DeviceProps, size: usize) -> Result<KernelSrc, IRTrace> {
        let builder = IRBuilder::default();

        let a = builder.add_input(8, DType::F32);
        let b = builder.add_input(8, DType::F32);
        let x = builder.add_input(size * 8, DType::F32);
        let y = ((a.broadcast([8], 0, size)? * x)? + b.broadcast([8, 1], 1, size)?)?;
        let ir = builder.build([x, y]);

        let sub = SubGraph::new(ir, vec![a.node(), b.node(), x.node()], vec![x.node(), y.node()])?;
        unsafe { super::generate(&sub, props)?.unwrap().0.lower("axby".to_string(), props).map_err(IRTrace::from) }
    }

    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    fn axby<G: Gpu>() -> Result<(), G::Error> {
        let device = Device::<G>::new(0)?;
        let stream = device.new_stream()?;

        let src = make_axby(device.props(), 4).unwrap();

        let axby = src.compile(device.clone())?;

        let aval = TValue::F32(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        let bval = TValue::F32(vec![4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0]);
        let xval = TValue::F32([4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0].repeat(4));

        let aval_buf = Buffer::from_host(&device, &aval)?;
        let bval_buf = Buffer::from_host(&device, &bval)?;
        let xval_buf = Buffer::from_host(&device, &xval)?;

        let x2val_buf = Buffer::zeroed(&device, xval.dtype(), xval.size())?;
        let yval_buf = Buffer::zeroed(&device, xval.dtype(), xval.size())?;

        axby.execute(stream.clone(), vec![aval_buf, bval_buf, xval_buf], vec![x2val_buf.clone(), yval_buf.clone()])?
            .value()?;

        let actualx = x2val_buf.to_host()?;
        let actualy = yval_buf.to_host()?;

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

    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    fn make_concat(props: &DeviceProps, dim: usize) -> Result<KernelSrc, IRTrace> {
        let builder = IRBuilder::default();

        let a = builder.add_input(8, DType::F32);
        let b = builder.add_input(8, DType::F32);

        let apad = a.pad([2, 2, 2], dim, 0, 2, DValue::F32(0.0))?;
        let bpad = b.pad([2, 2, 2], dim, 2, 0, DValue::F32(0.0))?;
        let concat = (apad + bpad)?;

        let ir = builder.build([concat]);

        let sub = SubGraph::new(ir, vec![a.node(), b.node()], vec![concat.node()])?;
        unsafe { super::generate(&sub, props)?.unwrap().0.lower("concat".to_string(), props).map_err(IRTrace::from) }
    }

    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    fn concat<G: Gpu>(dim: usize, expected: impl Into<Vec<f32>>) -> Result<(), G::Error> {
        let device = Device::<G>::new(0)?;
        let stream = device.new_stream()?;

        let src = make_concat(device.props(), dim).unwrap();

        let concat = src.compile(device.clone())?;

        let aval = TValue::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let bval = TValue::F32(vec![4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0]);

        let aval_buf = Buffer::from_host(&device, &aval)?;
        let bval_buf = Buffer::from_host(&device, &bval)?;

        let concat_buf = Buffer::zeroed(&device, DType::F32, 16)?;

        concat.execute(stream.clone(), vec![aval_buf, bval_buf], vec![concat_buf.clone()])?.value()?;

        assert_eq!(concat_buf.to_host()?, TValue::F32(expected.into()));

        Ok(())
    }

    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    fn make_slice(props: &DeviceProps, dim: usize) -> Result<KernelSrc, IRTrace> {
        let builder = IRBuilder::default();

        let input = builder.add_input(8, DType::F32);
        let slice = input.slice([2, 2, 2], dim, 0, 1)?;

        let ir = builder.build([slice]);

        let sub = SubGraph::new(ir, vec![input.node()], vec![slice.node()])?;
        unsafe { super::generate(&sub, props)?.unwrap().0.lower("slice".to_string(), props).map_err(IRTrace::from) }
    }

    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    fn slice<G: Gpu>(dim: usize, expected: impl Into<Vec<f32>>) -> Result<(), G::Error> {
        let device = Device::<G>::new(0)?;
        let stream = device.new_stream()?;

        let src = make_slice(device.props(), dim).unwrap();

        let slice = src.compile(device.clone())?;

        let input = TValue::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let input_buf = Buffer::from_host(&device, &input)?;
        let slice_buf = Buffer::zeroed(&device, DType::F32, 4)?;
        slice.execute(stream.clone(), vec![input_buf], vec![slice_buf.clone()])?.value()?;

        assert_eq!(slice_buf.to_host()?, TValue::F32(expected.into()));

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

        #[test]
        fn slice() -> Result<(), CudaError> {
            super::slice::<Cuda>(0, [1.0, 2.0, 3.0, 4.0])?;
            super::slice::<Cuda>(1, [1.0, 2.0, 5.0, 6.0])?;
            super::slice::<Cuda>(2, [1.0, 3.0, 5.0, 7.0])
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
            super::concat::<ROCm>(0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0])?;
            super::concat::<ROCm>(1, [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0])?;
            super::concat::<ROCm>(2, [1.0, 2.0, 4.0, 3.0, 3.0, 4.0, 2.0, 1.0, 5.0, 6.0, 1.0, 2.0, 7.0, 8.0, 3.0, 4.0])
        }

        #[test]
        fn slice() -> Result<(), ROCmError> {
            super::slice::<ROCm>(0, [1.0, 2.0, 3.0, 4.0])?;
            super::slice::<ROCm>(1, [1.0, 2.0, 5.0, 6.0])?;
            super::slice::<ROCm>(2, [1.0, 3.0, 5.0, 7.0])
        }
    }

    #[cfg(feature = "metal")]
    mod metal {
        use crate::runtime::metal::{Metal, MetalError};

        #[test]
        fn axby() -> Result<(), MetalError> {
            super::axby::<Metal>()
        }

        #[test]
        fn concat() -> Result<(), MetalError> {
            super::concat::<Metal>(
                0,
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0],
            )?;
            super::concat::<Metal>(
                1,
                [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0],
            )?;
            super::concat::<Metal>(2, [1.0, 2.0, 4.0, 3.0, 3.0, 4.0, 2.0, 1.0, 5.0, 6.0, 1.0, 2.0, 7.0, 8.0, 3.0, 4.0])
        }

        #[test]
        fn slice() -> Result<(), MetalError> {
            super::slice::<Metal>(0, [1.0, 2.0, 3.0, 4.0])?;
            super::slice::<Metal>(1, [1.0, 2.0, 5.0, 6.0])?;
            super::slice::<Metal>(2, [1.0, 3.0, 5.0, 7.0])
        }
    }
}
