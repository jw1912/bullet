mod args;
mod expr;

pub use args::{ConcreteKernelArgs, ConcreteKernelInput, KernelArgs, KernelInput};
pub use expr::Expr;

use acyclib::device::{OperationError, function::DeviceOperation};
use cudarc::{
    driver::{CudaFunction, LaunchConfig, PushKernelArg},
    nvrtc,
};

use crate::{CudaDevice, CudaError};

#[derive(Debug)]
pub struct Kernel {
    name: String,
    code: String,
    args: KernelArgs,
    func: CudaFunction,
}

impl Kernel {
    /// # Safety
    /// Must ensure that kernel will soundly execute with given args (which will be checked).
    pub unsafe fn new(name: String, code: String, args: KernelArgs) -> Result<Self, OperationError<CudaError>> {
        let ptx = nvrtc::compile_ptx(&code).map_err(CudaError::RuntimeCompile)?;
        let concrete = args.concretify()?;
        let module = concrete.device.stream().context().load_module(ptx).map_err(CudaError::Driver)?;
        let func = module.load_function("kernel").map_err(CudaError::Driver)?;

        drop(concrete);

        Ok(Self { name, code, args, func })
    }

    pub fn execute(&self) -> Result<(), OperationError<CudaError>> {
        let ConcreteKernelArgs { device, mut inputs, grid_dim, block_dim, shared_mem_bytes } =
            self.args.concretify()?;

        let stream = device.stream();
        let mut builder = &mut stream.launch_builder(&self.func);

        for input in &mut inputs {
            builder = match input {
                ConcreteKernelInput::F32(x) => builder.arg(&*x),
                ConcreteKernelInput::Size(x) => builder.arg(&*x),
                ConcreteKernelInput::SliceF32(x) => builder.arg(&**x),
                ConcreteKernelInput::SliceI32(x) => builder.arg(&**x),
                ConcreteKernelInput::MutSliceF32(x) => builder.arg(&mut **x),
                ConcreteKernelInput::MutSliceI32(x) => builder.arg(&mut **x),
            };
        }

        let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes };

        unsafe {
            builder.launch(cfg).map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    pub fn code(&self) -> String {
        self.code.clone()
    }
}

impl DeviceOperation<CudaDevice> for Kernel {
    fn opname(&self) -> String {
        self.name.clone()
    }

    fn execute(&self) -> Result<(), OperationError<CudaError>> {
        self.execute()
    }
}
