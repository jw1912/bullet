use bullet_core::{
    backend::device::base::{AdamConfig, BaseOperations},
    graph::ir::op::DiffableFromOutput,
};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{CudaBuffer, CudaError};

#[allow(unused)]
impl BaseOperations for CudaBuffer<f32> {
    type BaseError = CudaError;

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

        let threads = 512;
        let float4_size = (size as u32 + 3) / 4;
        let blocks = float4_size.div_ceil(threads);

        let func = self.device.module.load_function(func_name).map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&a.buf.slice(0..size))
                .arg(&mut self.buf.slice_mut(0..size))
                .launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 })
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

        let threads = 512;
        let float4_size = (size as u32 + 3) / 4;
        let blocks = float4_size.div_ceil(threads);

        let func = self.device.module.load_function(func_name).map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&a.buf.slice(0..size))
                .arg(&grd.buf.slice(0..size))
                .arg(&mut self.buf.slice_mut(0..size))
                .launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 })
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn add_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError> {
        Ok(())
    }

    fn abs_pow_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError> {
        Ok(())
    }

    fn abs_pow_scalar_backward(
        &mut self,
        size: usize,
        alpha: f32,
        input: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError> {
        Ok(())
    }

    fn pairwise_fwd(&mut self, size: usize, batch_size: usize, a: &Self) -> Result<(), Self::BaseError> {
        Ok(())
    }

    fn pairwise_bwd(&mut self, size: usize, batch_size: usize, a: &Self, grd: &Self) -> Result<(), Self::BaseError> {
        Ok(())
    }

    fn power_error_fwd(&mut self, power: f32, size: usize, a: &Self, b: &Self) -> Result<(), Self::BaseError> {
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
        Ok(())
    }

    fn clip(&mut self, size: usize, min: f32, max: f32) -> Result<(), Self::BaseError> {
        let threads = 1024;
        let float4_size = (size as u32 + 3) / 4;
        let blocks = float4_size.div_ceil(threads);

        let func = self.device.module.load_function("ClipKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&mut self.buf.slice_mut(0..size))
                .arg(&min)
                .arg(&max)
                .launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 })
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
        let threads = 1024;
        let float4_size = (size as u32 + 3) / 4;
        let blocks = float4_size.div_ceil(threads);

        let func = self.device.module.load_function("AdamKernel").map_err(CudaError::Driver)?;

        unsafe {
            self.device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&config.beta1)
                .arg(&config.beta2)
                .arg(&config.gradient_factor)
                .arg(&config.learning_rate)
                .arg(&config.denom)
                .arg(&mut self.buf.slice_mut(0..size))
                .arg(&mut mom.buf.slice_mut(0..size))
                .arg(&mut vel.buf.slice_mut(0..size))
                .arg(&grd.buf.slice(0..size))
                .launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 })
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}
