use std::sync::Arc;

use acyclib::device::{
    DeviceBuffer,
    operation::{AdamConfig, BaseOperations, DiffableFromOutput, GemmConfig},
    tensor::Shape,
};
use cudarc::{
    cublas::Gemm,
    driver::{CudaSlice, DriverError, PushKernelArg},
};

use crate::{CudaBuffer, CudaDevice, CudaError, convert_gemm_config};

pub(crate) fn set_to(
    device: Arc<CudaDevice>,
    slice: &mut CudaSlice<f32>,
    size: usize,
    val: f32,
) -> Result<(), DriverError> {
    let func = device.module().load_function("SetKernel")?;

    unsafe {
        device
            .stream()
            .launch_builder(&func)
            .arg(&mut slice.slice_mut(0..size))
            .arg(&(size as i32))
            .arg(&val)
            .launch(CudaDevice::elementwise_launch_params_single(size, 512))?;
    }

    Ok(())
}

#[allow(unused)]
impl BaseOperations for CudaBuffer<f32> {
    type BaseError = CudaError;

    fn set_to(&mut self, size: usize, val: f32) -> Result<(), Self::BaseError> {
        set_to(self.device.clone(), &mut self.buf, size, val).map_err(CudaError::Driver)
    }

    fn diffable_from_output_fwd(
        &mut self,
        size: usize,
        a: &Self,
        act: DiffableFromOutput,
    ) -> Result<(), Self::BaseError> {
        let func_name = match act {
            DiffableFromOutput::Identity => panic!("No-op!"),
            DiffableFromOutput::ReLU => "ForwardReluKernel",
            DiffableFromOutput::CReLU => "ForwardCreluKernel",
            DiffableFromOutput::SCReLU => "ForwardScreluKernel",
            DiffableFromOutput::SqrReLU => "ForwardSqrReluKernel",
            DiffableFromOutput::Sigmoid => "ForwardSigmoidKernel",
        };

        let func = self.device.module().load_function(func_name).map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&a.buf.slice(0..size))
                .arg(&mut self.buf.slice_mut(0..size))
                .launch(CudaDevice::elementwise_launch_params(size, 512))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn diffable_from_output_bwd(
        &mut self,
        size: usize,
        a: &Self,
        grd: &Self,
        act: DiffableFromOutput,
    ) -> Result<(), Self::BaseError> {
        let func_name = match act {
            DiffableFromOutput::Identity => panic!("No-op!"),
            DiffableFromOutput::ReLU => "BackwardReluKernel",
            DiffableFromOutput::CReLU => "BackwardCreluKernel",
            DiffableFromOutput::SCReLU => "BackwardScreluKernel",
            DiffableFromOutput::SqrReLU => "BackwardSqrReluKernel",
            DiffableFromOutput::Sigmoid => "BackwardSigmoidKernel",
        };

        let func = self.device.module().load_function(func_name).map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&a.buf.slice(0..size))
                .arg(&grd.buf.slice(0..size))
                .arg(&mut self.buf.slice_mut(0..size))
                .launch(CudaDevice::elementwise_launch_params(size, 512))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn mul_scalar(&mut self, size: usize, alpha: f32) -> Result<(), Self::BaseError> {
        let func = self.device.module().load_function("ScaleAssignKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&mut self.buf.slice_mut(0..size))
                .arg(&alpha)
                .launch(CudaDevice::elementwise_launch_params(size, 512))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn linear_comb(&mut self, size: usize, alpha: f32, beta: f32, input: &Self) -> Result<(), Self::BaseError> {
        let func = self.device.module().load_function("ScaleAddAssignKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&alpha)
                .arg(&mut self.buf.slice_mut(0..size))
                .arg(&beta)
                .arg(&input.buf.slice(0..size))
                .launch(CudaDevice::elementwise_launch_params(size, 512))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn linear_comb_splat(
        &mut self,
        size: usize,
        reps: usize,
        alpha: f32,
        beta: f32,
        input: &Self,
    ) -> Result<(), Self::BaseError> {
        let device = self.device.clone();

        device.with_ones(reps, |ones| {
            let cfg = GemmConfig::new(beta, alpha, Shape::new(size, 1), false, Shape::new(1, reps), false);
            let cfg = convert_gemm_config(&cfg).0;

            unsafe { self.device.blas().gemm(cfg, &input.buf, ones, &mut self.buf).map_err(CudaError::Blas) }
        })
    }

    fn reduce_across_batch(
        &mut self,
        size: usize,
        reps: usize,
        output_mul: f32,
        input_mul: f32,
        input: &Self,
    ) -> Result<(), Self::BaseError> {
        let device = self.device.clone();

        device.with_ones(reps, |ones| {
            let cfg = GemmConfig::new(input_mul, output_mul, Shape::new(size, reps), false, Shape::new(reps, 1), false);
            let cfg = convert_gemm_config(&cfg).0;

            unsafe { self.device.blas().gemm(cfg, &input.buf, ones, &mut self.buf).map_err(CudaError::Blas) }
        })
    }

    fn add_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError> {
        let func = self.device.module().load_function("AddScalarKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&alpha)
                .arg(&input.buf.slice(0..size))
                .arg(&mut self.buf.slice_mut(0..size))
                .launch(CudaDevice::elementwise_launch_params(size, 512))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn abs_pow_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError> {
        let func = self.device.module().load_function("AbsPowScalarKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&alpha)
                .arg(&input.buf.slice(0..size))
                .arg(&mut self.buf.slice_mut(0..size))
                .launch(CudaDevice::elementwise_launch_params(size, 512))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn abs_pow_scalar_backward(
        &mut self,
        size: usize,
        alpha: f32,
        input: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError> {
        let func = self.device.module().load_function("AbsPowScalarBackwardKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&alpha)
                .arg(&input.buf.slice(0..size))
                .arg(&grd.buf.slice(0..size))
                .arg(&mut self.buf.slice_mut(0..size))
                .launch(CudaDevice::elementwise_launch_params(size, 512))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn pairwise_fwd(
        &mut self,
        offset: usize,
        stride: usize,
        size: usize,
        batch_size: usize,
        a: &Self,
    ) -> Result<(), Self::BaseError> {
        let output_size = size / 2;

        if size % 2 != 0
            || size * batch_size > a.size()
            || stride * batch_size > self.size()
            || stride < output_size
            || stride < offset + output_size
        {
            println!("{size}, {batch_size}, {stride}, {offset}");
            return Err(CudaError::ExpectedIllegalAddressAccess);
        }

        let func = self.device.module().load_function("PairwiseMulKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(stride as i32))
                .arg(&(output_size as i32))
                .arg(&(batch_size as i32))
                .arg(&a.buf)
                .arg(&mut self.buf.slice_mut(offset..))
                .launch(CudaDevice::elementwise_launch_params_single(batch_size * output_size, 1024))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn pairwise_bwd(
        &mut self,
        offset: usize,
        stride: usize,
        size: usize,
        batch_size: usize,
        a: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError> {
        let output_size = size / 2;

        if size % 2 != 0
            || size * batch_size > a.size()
            || size * batch_size > self.size()
            || stride * batch_size > grd.size()
            || stride < output_size + offset
        {
            println!("{size}, {batch_size}, {stride}, {offset}");
            return Err(CudaError::ExpectedIllegalAddressAccess);
        }

        let func = self.device.module().load_function("PairwiseMulBackwardKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(stride as i32))
                .arg(&(output_size as i32))
                .arg(&(batch_size as i32))
                .arg(&a.buf)
                .arg(&grd.buf.slice(offset..))
                .arg(&mut self.buf)
                .launch(CudaDevice::elementwise_launch_params_single(batch_size * output_size, 1024))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn power_error_fwd(&mut self, power: f32, size: usize, a: &Self, b: &Self) -> Result<(), Self::BaseError> {
        let func = self.device.module().load_function("PowerErrorKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&a.buf.slice(0..size))
                .arg(&b.buf.slice(0..size))
                .arg(&mut self.buf.slice_mut(0..size))
                .arg(&power)
                .launch(CudaDevice::elementwise_launch_params_single(size, 512))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn power_error_bwd(
        &mut self,
        power: f32,
        size: usize,
        a: &Self,
        b: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError> {
        let func = self.device.module().load_function("PowerErrorBackwardKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&a.buf.slice(0..size))
                .arg(&b.buf.slice(0..size))
                .arg(&grd.buf.slice(0..size))
                .arg(&mut self.buf.slice_mut(0..size))
                .arg(&power)
                .launch(CudaDevice::elementwise_launch_params_single(size, 512))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn copy_or_add_strided(
        &mut self,
        add: bool,
        rows: usize,
        cols: usize,
        offset: usize,
        stride: usize,
        a: &Self,
        offset_a: usize,
        stride_a: usize,
    ) -> Result<(), Self::BaseError> {
        let func = self.device.module().load_function("LinearCombStridedKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&i32::from(add))
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(stride_a as i32))
                .arg(&(stride as i32))
                .arg(&a.buf.slice(offset_a..))
                .arg(&mut self.buf.slice_mut(offset..))
                .launch(CudaDevice::elementwise_launch_params_single(rows * cols, 512))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn clip(&mut self, size: usize, min: f32, max: f32) -> Result<(), Self::BaseError> {
        let func = self.device.module().load_function("ClipKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&mut self.buf.slice_mut(0..size))
                .arg(&min)
                .arg(&max)
                .launch(CudaDevice::elementwise_launch_params(size, 1024))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn adam(
        &mut self,
        config: &AdamConfig,
        size: usize,
        grd: &Self,
        mom: &mut Self,
        vel: &mut Self,
    ) -> Result<(), Self::BaseError> {
        let func = self.device.module().load_function("AdamKernel").map_err(CudaError::Driver)?;

        let (min, max) = config.clip.unwrap_or((1.0, 1.0));

        unsafe {
            self.device
                .stream()
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&config.beta1)
                .arg(&config.beta2)
                .arg(&config.gradient_factor)
                .arg(&config.learning_rate)
                .arg(&config.decay)
                .arg(&min)
                .arg(&max)
                .arg(&config.denom)
                .arg(&mut self.buf.slice_mut(0..size))
                .arg(&mut mom.buf.slice_mut(0..size))
                .arg(&mut vel.buf.slice_mut(0..size))
                .arg(&grd.buf.slice(0..size))
                .launch(CudaDevice::elementwise_launch_params(size, 1024))
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}
