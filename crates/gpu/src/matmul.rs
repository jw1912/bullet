use bullet_compiler::tensor::{
    DType, IRTrace, TensorIR,
    operation::{Matmul, MatrixLayout, ReduceAcrossDimension, Reduction},
    transform::IRTransform,
};

/// It seems rocBLAS has really bad GEMM performance on some matrix dimensions,
/// which can be restructured into batched GEMM calls to gain performance
#[derive(Clone, Debug)]
pub(crate) struct SplitK;

impl IRTransform for SplitK {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        for op in ir.operations() {
            let Some(mm) = op.data().downcast::<Matmul>().copied() else { continue };
            let k = mm.lhs.cols.get();
            let size = mm.lhs.rows * mm.rhs.cols;

            if mm.dtype != DType::F32
                || (!mm.lhs.col_mjr && mm.lhs.rows.get() != 1)
                || (mm.rhs.col_mjr && mm.rhs.cols.get() != 1)
                || k < 4096
                || size.get() > 65536
            {
                continue;
            }

            // Already well-populated grids do not need extra parallelism. Bound
            // temporary storage to 64 MiB even when the original GEMM is batched.
            let tiles = mm.batch.get() * mm.lhs.rows.get().div_ceil(64) * mm.rhs.cols.get().div_ceil(128);
            if tiles >= 64 {
                continue;
            }
            let mut splits = 1;
            while k.is_multiple_of(splits * 2)
                && k / (splits * 2) >= 512
                && splits < 256
                && mm.batch.get() * size.get() * splits * 2 <= 16 * 1024 * 1024
            {
                splits *= 2;
            }
            if splits == 1 {
                continue;
            }
            let chunk = (k / splits).into();
            let partial = Matmul::new(
                mm.dtype,
                mm.batch * splits,
                MatrixLayout { cols: chunk, col_mjr: true, ..mm.lhs },
                MatrixLayout { rows: chunk, col_mjr: false, ..mm.rhs },
            )?;
            let partial = ir.add_op(op.inputs(), Ok::<_, IRTrace>(partial))?[0];
            let reduce = ReduceAcrossDimension::new(mm.dtype, [mm.batch, splits.into(), size], 1, Reduction::Sum)?;
            ir.replace_operation(op.id(), [partial], reduce)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_compiler::{
        ir::NodeId,
        tensor::{IRBuilder, TValue},
    };

    /// `(batch, m, n, k, lhs col-major, rhs col-major)`
    type Case = (usize, usize, usize, usize, bool, bool);

    const SPLIT: [Case; 5] = [
        (2, 3, 5, 4096, true, false),
        (2, 8, 4, 4096, true, false),
        (1, 1, 5, 6144, false, false),
        (2, 3, 1, 4096, true, true),
        (1, 1, 1, 4096, true, true),
    ];

    const NO_SPLIT: [Case; 5] = [
        (1, 3, 5, 4096, false, false),
        (1, 3, 5, 4096, true, true),
        (1, 3, 5, 4097, true, false),
        (1, 3, 5, 2048, true, false),
        (64, 1, 1, 4096, true, false),
    ];

    fn build((batch, m, n, k, col_a, col_b): Case) -> (TensorIR, [(NodeId, TValue); 2]) {
        let builder = IRBuilder::default();
        let a = builder.add_input(batch * m * k, DType::F32);
        let b = builder.add_input(batch * k * n, DType::F32);
        let mm = Matmul::new(
            DType::F32,
            batch,
            MatrixLayout { rows: m.into(), cols: k.into(), col_mjr: col_a },
            MatrixLayout { rows: k.into(), cols: n.into(), col_mjr: col_b },
        )
        .unwrap();
        let out = builder.add_op([a, b], mm).unwrap()[0];
        // Small integers give exact FP32 sums in either accumulation order.
        let inputs = [
            (a.node(), TValue::F32((0..batch * m * k).map(|i| (i % 7) as f32 - 3.0).collect())),
            (b.node(), TValue::F32((0..batch * k * n).map(|i| (i % 11) as f32 - 5.0).collect())),
        ];
        (builder.build([out]), inputs)
    }

    fn compare(case: Case, split: bool) {
        let (mut ir, inputs) = build(case);
        let expected = ir.evaluate(inputs.clone()).unwrap().unwrap();
        ir.transform(SplitK).unwrap();
        ir.check_valid().unwrap();
        assert_eq!(ir.operations().iter().any(|op| op.data().downcast::<ReduceAcrossDimension>().is_some()), split);
        assert_eq!(ir.evaluate(inputs).unwrap().unwrap(), expected);
    }

    #[test]
    fn split_k_preserves_batched_products_and_degenerate_layouts() {
        SPLIT.into_iter().for_each(|case| compare(case, true));
    }

    #[test]
    fn split_k_keeps_unsupported_layouts_and_short_or_indivisible_reductions() {
        NO_SPLIT.into_iter().for_each(|case| compare(case, false));
    }

    /// Exercise the full lowering pipeline, including partial-product reduction where enabled.
    #[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
    fn compare_device<G: crate::runtime::Gpu>() -> Result<(), G::Error> {
        use crate::{buffer::Buffer, function::Function, runtime::Device};
        use std::collections::BTreeMap;

        let device = Device::<G>::new(0)?;
        let stream = device.new_stream()?;

        for case in SPLIT.into_iter().chain(NO_SPLIT) {
            let (ir, inputs) = build(case);
            let expected = ir.evaluate(inputs.clone()).unwrap().unwrap();

            let mut bufs = BTreeMap::new();
            for (id, value) in inputs {
                bufs.insert(id, Buffer::from_host(&device, &value)?);
            }
            for (&id, value) in &expected {
                bufs.insert(id, Buffer::zeroed(&device, value.dtype(), value.size())?);
            }

            let mut func = Function::new(device.clone(), ir).unwrap();
            func.prealloc()?;
            func.execute(stream.clone(), &bufs)?.value()?;
            for (&id, value) in &expected {
                assert_eq!(&bufs[&id].to_host()?, value, "{case:?}");
            }
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::runtime::cuda::{Cuda, CudaError};

        #[test]
        fn split_k_device() -> Result<(), CudaError> {
            super::compare_device::<Cuda>()
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use crate::runtime::rocm::{ROCm, ROCmError};

        #[test]
        fn split_k_device() -> Result<(), ROCmError> {
            super::compare_device::<ROCm>()
        }
    }

    #[cfg(feature = "metal")]
    mod metal {
        use crate::runtime::metal::{Metal, MetalError};

        #[test]
        fn split_k_device() -> Result<(), MetalError> {
            super::compare_device::<Metal>()
        }
    }
}
