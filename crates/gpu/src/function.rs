use std::{collections::HashMap, sync::Arc};

use bullet_compiler::tensor::{DValue, IRTrace, Size, TensorIR};

use crate::{
    buffer::{Buffer, SyncOnDrop, SyncOnValue},
    runtime::{Device, Dim3, Gpu, Kernel, Stream},
};

enum Arg {
    Pointer { idx: usize },
    Value(DValue),
    Size(Size),
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
        func: Kernel<G>,
        args: Vec<Arg>,
        gdim: Box<dyn Fn(usize) -> Dim3>,
        bdim: Box<dyn Fn(usize) -> u32>,
        smem: Box<dyn Fn(usize) -> u32>,
    },
}

pub struct Function<G: Gpu> {
    maps: HashMap<String, (usize, bool, Size)>,
    ptrs: Box<[G::DevicePtr]>,
    insts: Box<[Inst<G>]>,
}

impl<G: Gpu> Function<G> {
    pub fn new(device: Arc<Device<G>>, ir: TensorIR) -> Result<Self, IRTrace> {
        let maps = HashMap::new();
        let ptrs = Vec::new();
        let insts = Vec::new();

        for op in ir.ordered_operations()? {
            if op.data().is_input() {
                unimplemented!()
            }
        }

        Ok(Self { maps, ptrs: ptrs.into_boxed_slice(), insts: insts.into_boxed_slice() })
    }

    pub fn execute(
        &mut self,
        stream: Arc<Stream<G>>,
        inputs: &HashMap<String, Arc<Buffer<G>>>,
    ) -> Result<SyncOnValue<G, &mut Self>, G::Error> {
        let mut sync = SyncOnDrop::new(stream.clone());

        let mut ptrs = HashMap::new();
        let mut var_size = None;

        for (name, buf) in inputs {
            let (idx, is_mut, size) = *self.maps.get(name).ok_or("Input not in function!".into())?;

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
            self.ptrs[idx] = ptr;

            if let Some(is_alr_mut) = ptrs.insert(ptr, is_mut) {
                if is_mut || is_alr_mut {
                    return Err("Cannot alias pointers!".to_string().into());
                }
            }
        }

        let var = var_size.unwrap_or(1);

        unsafe {
            for inst in &mut self.insts {
                match inst {
                    &mut Inst::Memset { idx, bytes, value } => {
                        let bytes = bytes.evaluate(var);
                        stream.memset(self.ptrs[idx], bytes, value)?;
                    }
                    &mut Inst::Malloc { idx, bytes } => {
                        let bytes = bytes.evaluate(var);
                        self.ptrs[idx] = stream.malloc(bytes)?;
                    }
                    &mut Inst::Free { idx } => {
                        stream.free(self.ptrs[idx])?;
                    }
                    Inst::LaunchKernel { func, args, gdim, bdim, smem } => {
                        let mut sizes = Vec::new();
                        let mut args: Vec<_> = args
                            .iter()
                            .map(|arg| match arg {
                                Arg::Pointer { idx } => args.as_ptr().add(*idx).cast_mut().cast(),
                                Arg::Value(val) => val.ptr(),
                                Arg::Size(size) => {
                                    sizes.push(size.evaluate(var) as i32);
                                    (sizes.last().unwrap() as *const i32).cast_mut().cast()
                                }
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
