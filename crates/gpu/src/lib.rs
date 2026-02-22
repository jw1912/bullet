//! Crate for compiling and executing tensor DAGs from `bullet-compiler` on CUDA/ROCm devices.

pub mod buffer;
pub mod function;
pub mod operations;
pub mod runtime;

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[cfg(test)]
mod tests {
    use bullet_compiler::{
        ir::NodeId,
        tensor::{DType, IRBuilder, IRTrace, Size, TValue, TensorIR},
    };

    use crate::{
        buffer::Buffer,
        function::Function,
        operations::pointwise::LowerPointwise,
        runtime::{Device, Gpu},
    };

    fn make_axby() -> Result<(TensorIR, [NodeId; 4]), IRTrace> {
        let size = Size::variable();

        let builder = IRBuilder::default();

        let a = builder.add_input(8, DType::F32);
        let b = builder.add_input(1, DType::F32);

        let x = builder.add_input(size * 8, DType::F32);

        let y = ((a.broadcast([8], 0, size)? * x)? + b.broadcast([1], 0, size * 8)?)?;

        let mut ir = builder.build([y]);

        ir.optimise()?;

        Ok((ir, [a.node(), b.node(), x.node(), y.node()]))
    }

    fn axby<G: Gpu>() -> Result<(), G::Error> {
        let (mut ir, [a, b, x, y]) = make_axby().unwrap();

        ir.transform(LowerPointwise).unwrap();

        let device = Device::<G>::new(0)?;
        let stream = device.clone().new_stream()?;

        let mut func = Function::new(device, ir).unwrap();

        let buf_a =
            Buffer::from_host(stream.clone(), &TValue::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))?.value().0;
        let buf_b = Buffer::from_host(stream.clone(), &TValue::F32(vec![1.0]))?.value().0;
        let buf_x =
            Buffer::from_host(stream.clone(), &TValue::F32([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0].repeat(16)))?
                .value()
                .0;
        let buf_y = Buffer::from_host(stream.clone(), &TValue::F32(vec![0.0; 8 * 16]))?.value().0;

        let sync = func.execute(
            stream.clone(),
            &[(a, buf_a.clone()), (b, buf_b.clone()), (x, buf_x.clone()), (y, buf_y.clone())].into(),
        )?;

        drop(sync);

        assert_eq!(
            buf_y.clone().to_host(stream.clone())?.value(),
            TValue::F32([9.0, 15.0, 19.0, 21.0, 21.0, 19.0, 15.0, 9.0].repeat(16))
        );

        assert!(
            func.execute(
                stream.clone(),
                &[(a, buf_b.clone()), (b, buf_a.clone()), (x, buf_x.clone()), (y, buf_y.clone())].into()
            )
            .is_err()
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::runtime::cuda::{Cuda, CudaError};

        #[test]
        fn axby() -> Result<(), CudaError> {
            super::axby::<Cuda>()
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use crate::runtime::rocm::{ROCm, ROCmError};

        #[test]
        fn axby() -> Result<(), ROCmError> {
            super::axby::<ROCm>()
        }
    }
}
