use std::{collections::HashMap, sync::Arc};

use bullet_compiler::{
    graph::{DValue, Size},
    ir::{IR, IRTrace},
};

use crate::{
    buffer::{GpuBuffer, SyncOnDrop, SyncOnValue},
    runtime::{Dim3, Gpu, GpuDevice, GpuKernel, GpuStream},
};

enum Arg {
    Pointer { idx: usize },
    Value(DValue),
}

enum Inst<G: Gpu> {
    Memset {
        idx: usize,
        bytes: Size,
        value: u8,
    },
    Malloc {
        idx: usize,
        bytes: Size,
    },
    Free {
        idx: usize,
    },
    LaunchKernel {
        func: GpuKernel<G>,
        args: Vec<Arg>,
        gdim: Box<dyn Fn(usize) -> Dim3>,
        bdim: Box<dyn Fn(usize) -> Dim3>,
        smem: Box<dyn Fn(usize) -> usize>,
    },
}

pub struct GpuFunction<G: Gpu> {
    mappings: HashMap<String, (usize, bool, Size)>,
    pointers: Vec<G::DevicePtr>,
    instructions: Vec<Inst<G>>,
}

impl<G: Gpu> GpuFunction<G> {
    pub fn new(device: Arc<GpuDevice<G>>, _ir: IR) -> Result<Self, IRTrace> {
        let mappings = HashMap::new();
        let pointers = Vec::new();
        let instructions = Vec::new();

        Ok(Self { mappings, pointers, instructions })
    }

    pub fn execute(
        &mut self,
        stream: Arc<GpuStream<G>>,
        inputs: &HashMap<String, Arc<GpuBuffer<G>>>,
    ) -> Result<SyncOnValue<G, &mut Self>, G::Error> {
        let mut sync = SyncOnDrop::new(stream.clone());

        let mut ptrs = HashMap::new();
        let mut var_size = None;

        for (name, buf) in inputs {
            let (idx, is_mut, size) = *self.mappings.get(name).ok_or("Input not in function!".into())?;

            if let Some(new_var) = size.get_var_size(buf.size()) {
                if size.evaluate(new_var) != buf.size() {
                    return Err("Mismatched sizes!".to_string().into());
                }

                match var_size {
                    None => var_size = Some(new_var),
                    Some(old_var) => {
                        if old_var != new_var {
                            return Err("Mismatched var sizes!".to_string().into());
                        }
                    }
                }
            } else {
                match size.evaluate_constant() {
                    None => return Err("Invalid var size!".to_string().into()),
                    Some(len) => {
                        if len != buf.size() {
                            return Err("Mismatched sizes!".to_string().into());
                        }
                    }
                }
            }

            let guard = buf.clone().acquire(stream.clone())?;
            let ptr = guard.ptr();
            sync.attach(guard)?;
            self.pointers[idx] = ptr;

            if let Some(is_alr_mut) = ptrs.insert(ptr, is_mut) {
                if is_mut || is_alr_mut {
                    return Err("Cannot alias pointers!".to_string().into());
                }
            }
        }

        let var = var_size.unwrap_or(1);

        unsafe {
            for inst in &mut self.instructions {
                match inst {
                    &mut Inst::Memset { idx, bytes, value } => {
                        let bytes = bytes.evaluate(var);
                        stream.memset(self.pointers[idx], bytes, value)?;
                    }
                    &mut Inst::Malloc { idx, bytes } => {
                        let bytes = bytes.evaluate(var);
                        self.pointers[idx] = stream.malloc(bytes)?;
                    }
                    &mut Inst::Free { idx } => {
                        stream.free(self.pointers[idx])?;
                    }
                    Inst::LaunchKernel { func, args, gdim, bdim, smem } => {
                        let mut args: Vec<_> = args
                            .iter()
                            .map(|arg| match arg {
                                Arg::Pointer { idx } => args.as_ptr().add(*idx).cast_mut().cast(),
                                Arg::Value(val) => val.ptr(),
                            })
                            .collect();

                        func.launch(&stream, gdim(var), bdim(var), args.as_mut_ptr(), smem(var))?;
                    }
                }
            }
        }

        Ok(SyncOnValue::new(sync, self))
    }
}
