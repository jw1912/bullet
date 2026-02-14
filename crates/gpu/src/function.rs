use std::{collections::HashMap, ffi::c_void, sync::Arc};

use bullet_compiler::{
    graph::Size,
    ir::{IR, IRTrace},
};

use crate::{
    buffer::{GpuBuffer, SyncOnDrop, SyncOnValue},
    device::{Dim3, Gpu, GpuDevice, GpuStream},
};

trait Arg {}
impl Arg for *mut c_void {}
impl Arg for i32 {}
impl Arg for f32 {}

enum Inst {
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
        func: *const c_void,
        args: Vec<Box<dyn Arg>>,
        gdim: Box<dyn Fn(usize) -> Dim3>,
        bdim: Box<dyn Fn(usize) -> Dim3>,
        smem: Box<dyn Fn(usize) -> usize>,
    },
}

pub struct GpuFunction<G: Gpu> {
    device: Arc<GpuDevice<G>>,
    mappings: HashMap<String, (usize, bool, Size)>,
    pointers: Vec<*mut c_void>,
    instructions: Vec<Inst>,
}

impl<G: Gpu> GpuFunction<G> {
    pub fn new(device: Arc<GpuDevice<G>>, _ir: IR) -> Result<Self, IRTrace> {
        let mappings = HashMap::new();
        let pointers = Vec::new();
        let instructions = Vec::new();

        Ok(Self { device, mappings, pointers, instructions })
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

            match (var_size, size.get_var_size(buf.size())) {
                (_, None) => {}
                (Some(x), Some(y)) => {
                    if x != y {
                        return Err(format!("Mismatched var sizes: {x} != {y}!").into());
                    }
                }
                (None, Some(size)) => var_size = Some(size),
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
                        stream.malloc(&mut self.pointers[idx], bytes)?;
                    }
                    &mut Inst::Free { idx } => {
                        stream.free(self.pointers[idx])?;
                    }
                    Inst::LaunchKernel { func, args, gdim, bdim, smem } => {
                        let mut args: Vec<_> =
                            args.iter_mut().map(|arg| arg.as_mut() as *mut dyn Arg as *mut c_void).collect();

                        stream.launch_kernel(*func, gdim(var), bdim(var), args.as_mut_ptr(), smem(var))?;
                    }
                }
            }
        }

        Ok(SyncOnValue::new(sync, self))
    }
}
