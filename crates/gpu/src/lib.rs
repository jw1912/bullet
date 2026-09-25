//! Crate for compiling and executing tensor DAGs from `bullet-compiler` on CUDA/ROCm devices.

pub mod buffer;
pub mod function;
pub mod kernel;
pub mod pointwise;
pub mod runtime;

#[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
#[cfg(test)]
mod tests {
    use bullet_compiler::{
        ir::NodeId,
        tensor::{DType, IRBuilder, IRTrace, TValue, TensorIR},
    };

    use crate::{
        buffer::Buffer,
        function::Function,
        runtime::{Device, Gpu},
    };

    fn make_axby(size: usize) -> Result<(TensorIR, [NodeId; 6]), IRTrace> {
        let builder = IRBuilder::default();

        let a = builder.add_input(8, DType::F32);
        let b = builder.add_input(1, DType::F32);

        let x = builder.add_input(size * 8, DType::F32);

        let y = ((a.broadcast([8], 0, size)? * x)? + b.broadcast([1], 0, size * 8)?)?;
        let z = y.reduce_max([size, 8], 1)?;
        let w = y.reduce_sum([size, 8], 0)?;

        let mut ir = builder.build([y, z, w]);

        ir.optimise()?;

        Ok((ir, [a.node(), b.node(), x.node(), y.node(), z.node(), w.node()]))
    }

    fn axby<G: Gpu>() -> Result<(), G::Error> {
        let batch_size = 256;

        let (ir, [a, b, x, y, z, w]) = make_axby(batch_size).unwrap();

        let device = Device::<G>::new(0)?;
        let stream = device.new_stream()?;

        let mut func = Function::new(device.clone(), ir).unwrap();
        func.prealloc().unwrap();

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

    fn reduce_max_min_i32<G: Gpu>() -> Result<(), G::Error> {
        let builder = IRBuilder::default();
        let x = builder.add_input(8, DType::I32);
        let mx = x.reduce_max([2, 4], 1).unwrap();
        let mn = x.reduce_min([2, 4], 1).unwrap();
        let mut ir = builder.build([mx, mn]);
        ir.optimise().unwrap();

        let device = Device::<G>::new(0)?;
        let stream = device.new_stream()?;

        let mut func = Function::new(device.clone(), ir).unwrap();
        func.prealloc().unwrap();

        let buf_x = Buffer::from_host(&device, &TValue::I32(vec![-1, -5, -2, -3, 7, -8, 0, 3]))?;
        let buf_mx = Buffer::from_host(&device, &TValue::I32(vec![0; 2]))?;
        let buf_mn = Buffer::from_host(&device, &TValue::I32(vec![0; 2]))?;

        func.execute(
            stream.clone(),
            &[(x.node(), buf_x), (mx.node(), buf_mx.clone()), (mn.node(), buf_mn.clone())].into(),
        )?
        .value()?;

        assert_eq!(buf_mx.to_host()?, TValue::I32(vec![-1, 7]));
        assert_eq!(buf_mn.to_host()?, TValue::I32(vec![-5, -8]));

        Ok(())
    }

    fn pad_non_finite<G: Gpu>() -> Result<(), G::Error> {
        let builder = IRBuilder::default();
        let x = builder.add_input(2, DType::F32);
        let y = x.pad([2], 0, 1, 0, f32::NEG_INFINITY.into()).unwrap();
        let z = x.pad([2], 0, 0, 1, f32::NAN.into()).unwrap();
        let mut ir = builder.build([y, z]);
        ir.optimise().unwrap();

        let device = Device::<G>::new(0)?;
        let stream = device.new_stream()?;

        let mut func = Function::new(device.clone(), ir).unwrap();
        func.prealloc().unwrap();

        let buf_x = Buffer::from_host(&device, &TValue::F32(vec![1.0, 2.0]))?;
        let buf_y = Buffer::from_host(&device, &TValue::F32(vec![0.0; 3]))?;
        let buf_z = Buffer::from_host(&device, &TValue::F32(vec![0.0; 3]))?;

        func.execute(
            stream.clone(),
            &[(x.node(), buf_x), (y.node(), buf_y.clone()), (z.node(), buf_z.clone())].into(),
        )?
        .value()?;

        assert_eq!(buf_y.to_host()?, TValue::F32(vec![f32::NEG_INFINITY, 1.0, 2.0]));

        let TValue::F32(z) = buf_z.to_host()? else { panic!() };
        assert_eq!(&z[..2], &[1.0, 2.0]);
        assert!(z[2].is_nan());

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
        fn reduce_max_min_i32() -> Result<(), CudaError> {
            super::reduce_max_min_i32::<Cuda>()
        }

        #[test]
        fn pad_non_finite() -> Result<(), CudaError> {
            super::pad_non_finite::<Cuda>()
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
        fn reduce_max_min_i32() -> Result<(), ROCmError> {
            super::reduce_max_min_i32::<ROCm>()
        }

        #[test]
        fn pad_non_finite() -> Result<(), ROCmError> {
            super::pad_non_finite::<ROCm>()
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
        fn reduce_max_min_i32() -> Result<(), MetalError> {
            super::reduce_max_min_i32::<Metal>()
        }

        #[test]
        fn pad_non_finite() -> Result<(), MetalError> {
            super::pad_non_finite::<Metal>()
        }
    }
}
