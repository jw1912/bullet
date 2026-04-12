//! Crate for compiling and executing tensor DAGs from `bullet-compiler` on CUDA/ROCm devices.

pub mod buffer;
pub mod function;
pub mod kernel;
pub mod pointwise;
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
        runtime::{Device, Gpu},
    };

    fn make_axby() -> Result<(TensorIR, [NodeId; 6]), IRTrace> {
        let size = Size::variable();

        let builder = IRBuilder::default();

        let a = builder.add_input(8, DType::F32);
        let b = builder.add_input(1, DType::F32);

        let x = builder.add_input(size * 8, DType::F32);

        let y = ((a.broadcast([8], 0, size)? * x)? + b.broadcast([1], 0, size * 8)?)?;
        let z = y.reduce_max([size, 8.into()], 1)?;
        let w = y.reduce_sum([size, 8.into()], 0)?;

        let mut ir = builder.build([y, z, w]);

        ir.optimise()?;

        Ok((ir, [a.node(), b.node(), x.node(), y.node(), z.node(), w.node()]))
    }

    fn axby<G: Gpu>() -> Result<(), G::Error> {
        let batch_size = 256;

        let (ir, [a, b, x, y, z, w]) = make_axby().unwrap();

        let device = Device::<G>::new(0)?;
        let stream = device.new_stream()?;

        let mut func = Function::new(device.clone(), ir).unwrap();
        func.prealloc(batch_size).unwrap();

        let buf_a = Buffer::from_host(&device, &TValue::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))?;
        let buf_b = Buffer::from_host(&device, &TValue::F32(vec![2.0]))?;
        let buf_x =
            Buffer::from_host(&device, &TValue::F32([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0].repeat(batch_size)))?;

        let buf_y = Buffer::from_host(&device, &TValue::F32(vec![10.0; 8 * batch_size]))?;
        let buf_z = Buffer::from_host(&device, &TValue::F32(vec![10.0; batch_size]))?;
        let buf_w = Buffer::from_host(&device, &TValue::F32(vec![10.0; 8]))?;

        func.execute(
            stream.clone(),
            &[
                (a, buf_a.clone()),
                (b, buf_b.clone()),
                (x, buf_x.clone()),
                (y, buf_y.clone()),
                (z, buf_z.clone()),
                (w, buf_w.clone()),
            ]
            .into(),
        )?
        .value()?;

        assert_eq!(
            buf_y.clone().to_host()?,
            TValue::F32([10.0, 16.0, 20.0, 22.0, 22.0, 20.0, 16.0, 10.0].repeat(batch_size))
        );

        assert_eq!(
            buf_w.clone().to_host()?,
            TValue::F32(
                [10.0, 16.0, 20.0, 22.0, 22.0, 20.0, 16.0, 10.0].iter().map(|x| batch_size as f32 * x).collect()
            )
        );

        assert_eq!(buf_z.clone().to_host()?, TValue::F32([22.0].repeat(batch_size)));

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
