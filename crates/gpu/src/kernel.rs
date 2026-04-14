use std::{collections::BTreeSet, ffi::c_void, fmt, rc::Rc, sync::Arc};

use bullet_compiler::tensor::{OpType, TType};

use crate::{
    buffer::{Buffer, BufferGuard, SyncOnDrop, SyncOnValue},
    runtime::{Device, Dim3, Gpu, Kernel, Module, Stream},
};

#[derive(Clone)]
pub struct KernelSrc {
    pub(crate) inputs: Vec<TType>,
    pub(crate) outputs: Vec<TType>,
    pub(crate) name: String,
    pub(crate) source: String,
    pub(crate) requires_var_size_arg: bool,
    pub(crate) arg_order: Vec<(usize, bool)>,
    pub(crate) requires_zero: BTreeSet<usize>,
    pub(crate) gdim: Rc<dyn Fn(usize) -> Dim3>,
    pub(crate) bdim: Rc<dyn Fn(usize) -> u32>,
    pub(crate) smem: Rc<dyn Fn(usize) -> u32>,
}

impl fmt::Debug for KernelSrc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu.kernel.source")
    }
}

impl KernelSrc {
    /// ### Safety
    ///
    /// I solemnly swear that as long as the passed input and output
    /// tensors to the compiled function have the correct TType and
    /// the variable size is passed correctly, then this kernel will
    /// not invoke UB.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        inputs: Vec<TType>,
        outputs: Vec<TType>,
        name: String,
        source: String,
        requires_var_size_arg: bool,
        arg_order: Vec<(usize, bool)>,
        requires_zero: BTreeSet<usize>,
        gdim: Rc<dyn Fn(usize) -> Dim3>,
        bdim: Rc<dyn Fn(usize) -> u32>,
        smem: Rc<dyn Fn(usize) -> u32>,
    ) -> Self {
        assert_eq!(arg_order.len(), inputs.len() + outputs.len());
        assert_eq!(
            inputs.len(),
            arg_order.iter().filter_map(|(idx, input)| input.then_some(*idx)).collect::<BTreeSet<_>>().len()
        );
        assert_eq!(
            outputs.len(),
            arg_order.iter().filter_map(|(idx, input)| (!input).then_some(*idx)).collect::<BTreeSet<_>>().len()
        );

        Self { inputs, outputs, name, source, requires_var_size_arg, arg_order, requires_zero, gdim, bdim, smem }
    }

    pub fn compile<G: Gpu>(&self, device: Arc<Device<G>>) -> Result<CompiledKernel<G>, G::Error> {
        let kernel = Module::new(device, &self.source)?.get_kernel(&self.name)?;

        Ok(CompiledKernel {
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            kernel,
            requires_var_size_arg: self.requires_var_size_arg,
            arg_order: self.arg_order.clone(),
            requires_zero: self.requires_zero.clone(),
            gdim: self.gdim.clone(),
            bdim: self.bdim.clone(),
            smem: self.smem.clone(),
        })
    }
}

impl OpType for KernelSrc {
    fn opname(&self) -> String {
        format!("gpu.rtc.{}", self.name)
    }

    fn inputs(&self) -> Vec<TType> {
        self.inputs.clone()
    }

    fn outputs(&self) -> Vec<TType> {
        self.outputs.clone()
    }
}

pub struct CompiledKernel<G: Gpu> {
    pub(crate) inputs: Vec<TType>,
    pub(crate) outputs: Vec<TType>,
    pub(crate) kernel: Kernel<G>,
    pub(crate) requires_var_size_arg: bool,
    pub(crate) arg_order: Vec<(usize, bool)>,
    pub(crate) requires_zero: BTreeSet<usize>,
    pub(crate) gdim: Rc<dyn Fn(usize) -> Dim3>,
    pub(crate) bdim: Rc<dyn Fn(usize) -> u32>,
    pub(crate) smem: Rc<dyn Fn(usize) -> u32>,
}

impl<G: Gpu> fmt::Debug for CompiledKernel<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu.kernel.compiled")
    }
}

impl<G: Gpu> CompiledKernel<G> {
    /// ### Safety
    ///
    /// I solemnly swear that as long as the passed input and output
    /// tensors to the kernel have the correct TType and the variable
    /// size is passed correctly, then this kernel will not invoke UB.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        inputs: Vec<TType>,
        outputs: Vec<TType>,
        kernel: Kernel<G>,
        requires_var_size_arg: bool,
        arg_order: Vec<(usize, bool)>,
        requires_zero: BTreeSet<usize>,
        gdim: Rc<dyn Fn(usize) -> Dim3>,
        bdim: Rc<dyn Fn(usize) -> u32>,
        smem: Rc<dyn Fn(usize) -> u32>,
    ) -> Self {
        assert_eq!(arg_order.len(), inputs.len() + outputs.len());
        assert_eq!(
            inputs.len(),
            arg_order.iter().filter_map(|(idx, input)| input.then_some(*idx)).collect::<BTreeSet<_>>().len()
        );
        assert_eq!(
            outputs.len(),
            arg_order.iter().filter_map(|(idx, input)| (!input).then_some(*idx)).collect::<BTreeSet<_>>().len()
        );

        Self { inputs, outputs, kernel, requires_var_size_arg, arg_order, requires_zero, gdim, bdim, smem }
    }

    pub fn execute(
        &self,
        stream: Arc<Stream<G>>,
        inputs: Vec<Arc<Buffer<G>>>,
        outputs: Vec<Arc<Buffer<G>>>,
    ) -> Result<SyncOnValue<G, &Self>, G::Error> {
        let mut sync = SyncOnDrop::new(stream.clone());

        let inputs =
            inputs.iter().map(|i| i.clone().acquire(stream.clone())).collect::<Result<Vec<BufferGuard<G>>, _>>()?;
        let outputs =
            outputs.iter().map(|o| o.clone().acquire(stream.clone())).collect::<Result<Vec<BufferGuard<G>>, _>>()?;

        let mut vars = BTreeSet::new();

        if inputs.len() != self.inputs.len() || outputs.len() != self.outputs.len() {
            return Err("Mismatched number of inputs/outputs!".to_string().into());
        }

        for (buf, &ttype) in inputs.iter().zip(&self.inputs).chain(outputs.iter().zip(&self.outputs)) {
            if buf.dtype() != ttype.dtype() {
                return Err("Mismatched dtypes!".to_string().into());
            }

            let concrete_size = buf.size();
            if let Some(var) = ttype.size().get_var_size(concrete_size) {
                vars.insert(var);
            } else if ttype.size().evaluate_constant().unwrap() != concrete_size {
                return Err("Mismatched sizes!".to_string().into());
            }
        }

        let var = match vars.len() {
            0 => 1,
            1 => *vars.iter().next().unwrap(),
            _ => return Err(format!("Mismatching batch sizes in inputs: {vars:?}").into()),
        };

        let mut args: Vec<*mut c_void> = Vec::new();

        let size = var as i32;
        if self.requires_var_size_arg {
            args.push((&size as *const i32).cast_mut().cast());
        }

        let mut ptrs = vec![Default::default(); self.arg_order.len()];
        for (i, &(index, is_input)) in self.arg_order.iter().enumerate() {
            ptrs[i] = if is_input { inputs[index].ptr() } else { outputs[index].ptr() };
            args.push((&ptrs[i] as *const G::DevicePtr).cast_mut().cast());
        }

        unsafe {
            if !self.requires_zero.is_empty() {
                unimplemented!();
            }

            self.kernel.launch(
                &stream,
                (self.gdim)(var),
                (self.bdim)(var),
                args.as_ptr().cast_mut().cast(),
                (self.smem)(var),
            )?;
        }

        for i in inputs {
            sync.attach(i)?;
        }

        for o in outputs {
            sync.attach(o)?;
        }

        Ok(SyncOnValue::new(sync, self))
    }
}

impl<G: Gpu> OpType for CompiledKernel<G> {
    fn opname(&self) -> String {
        "gpu.kernel.compiled".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        self.inputs.clone()
    }

    fn outputs(&self) -> Vec<TType> {
        self.outputs.clone()
    }
}
