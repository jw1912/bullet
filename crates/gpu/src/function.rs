use std::{collections::HashMap, rc::Rc, sync::Arc};

use bullet_compiler::{
    ir::NodeId,
    tensor::{IRTrace, Size, TType, TensorIR},
};

use crate::{
    buffer::{Buffer, SyncOnDrop, SyncOnValue},
    kernel::KernelSrc,
    runtime::{Device, Dim3, Gpu, Kernel, Module, Stream},
};

enum Arg {
    Pointer { idx: usize },
    Size(Size),
}

enum Inst<G: Gpu> {
    Malloc {
        idx: usize,
        ty: TType,
    },
    Free {
        idx: usize,
    },
    Zero {
        idx: usize,
        ty: TType,
    },
    LaunchKernel {
        func: Kernel<G>,
        args: Vec<Arg>,
        gdim: Rc<dyn Fn(usize) -> Dim3>,
        bdim: Rc<dyn Fn(usize) -> u32>,
        smem: Rc<dyn Fn(usize) -> u32>,
    },
}

pub struct Function<G: Gpu> {
    maps: HashMap<NodeId, (usize, bool, TType)>,
    insts: Box<[Inst<G>]>,
    num_ptrs: usize,
}

impl<G: Gpu> Function<G> {
    pub fn new(device: Arc<Device<G>>, ir: TensorIR) -> Result<Self, IRTrace> {
        let mut maps = HashMap::new();
        let mut num_ptrs = 0;
        let mut insts = Vec::new();

        let mut times_seen = HashMap::new();
        let mut indices = HashMap::new();

        for op in ir.ordered_operations()? {
            // allocate output buffers
            for &output in op.outputs() {
                if ir.is_input(output)? {
                    let input = op.outputs()[0];
                    maps.insert(input, (num_ptrs, false, ir.get_node(input)?.ty()));
                } else if ir.is_output(output) {
                    maps.insert(output, (num_ptrs, true, ir.get_node(output)?.ty()));
                } else {
                    insts.push(Inst::Malloc { idx: num_ptrs, ty: ir.get_node(output)?.ty() });
                    times_seen.insert(output, 0);
                }

                indices.insert(output, num_ptrs);
                num_ptrs += 1;
            }

            // insert kernels
            let data = op.data();
            if let Some(KernelSrc {
                source, requires_var_size_arg, arg_order, gdim, bdim, smem, requires_zero, ..
            }) = data.downcast().cloned()
            {
                let mut args = Vec::new();

                if requires_var_size_arg {
                    args.push(Arg::Size(Size::variable()));
                }

                for (index, is_input) in arg_order {
                    let node_id = if is_input { op.inputs()[index] } else { op.outputs()[index] };
                    args.push(Arg::Pointer { idx: *indices.get(&node_id).unwrap() });
                }

                for output in requires_zero {
                    let node_id = op.outputs()[output];
                    let ty = ir.get_node(node_id)?.ty();
                    insts.push(Inst::Zero { idx: *indices.get(&node_id).unwrap(), ty });
                }

                let func = Module::new(device.clone(), source)
                    .map_err(|e| IRTrace::from(format!("{e:?}")))?
                    .get_kernel("kernel")
                    .map_err(|e| IRTrace::from(format!("{e:?}")))?;

                insts.push(Inst::LaunchKernel { func, args, gdim, bdim, smem });
            } else if !data.is_input() {
                return Err(format!("Unsupported operation: {data:?}").into());
            }

            // free buffers that see no more usage
            for &input in op.inputs() {
                if !ir.is_input(input)? {
                    let times_seen = times_seen.get_mut(&input).unwrap();
                    *times_seen += 1;

                    if ir.get_node(input)?.children() == *times_seen {
                        let idx = *indices.get(&input).unwrap();
                        insts.push(Inst::Free { idx });
                    }
                }
            }
        }

        Ok(Self { maps, insts: insts.into_boxed_slice(), num_ptrs })
    }

    pub fn execute(
        &mut self,
        stream: Arc<Stream<G>>,
        inputs: &HashMap<NodeId, Arc<Buffer<G>>>,
    ) -> Result<SyncOnValue<G, &mut Self>, G::Error> {
        let mut sync = SyncOnDrop::new(stream.clone());

        let mut ptrs = vec![G::DevicePtr::default(); self.num_ptrs];

        let mut mutmap = HashMap::new();
        let mut var_size = None;

        for (name, buf) in inputs {
            let (idx, is_mut, ty) = *self.maps.get(name).ok_or("Input not in function!".into())?;
            let size = ty.size();

            if buf.dtype() != ty.dtype() {
                return Err("Mismatched dtypes!".to_string().into());
            }

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
            ptrs[idx] = ptr;

            if let Some(is_alr_mut) = mutmap.insert(ptr, is_mut) {
                if is_mut || is_alr_mut {
                    return Err("Cannot alias pointers!".to_string().into());
                }
            }
        }

        let var = var_size.unwrap_or(1);
        let mut sizes = Vec::new();

        unsafe {
            for inst in &mut self.insts {
                match inst {
                    &mut Inst::Malloc { idx, ty } => {
                        let bytes = ty.size().evaluate(var) * ty.dtype().bytes();
                        ptrs[idx] = stream.malloc(bytes)?;
                    }
                    &mut Inst::Free { idx } => {
                        stream.free(ptrs[idx])?;
                    }
                    &mut Inst::Zero { idx, ty } => {
                        let bytes = ty.size().evaluate(var) * ty.dtype().bytes();
                        stream.memset(ptrs[idx], bytes, 0)?;
                    }
                    Inst::LaunchKernel { func, args, gdim, bdim, smem } => {
                        sizes.clear();
                        let mut args: Vec<_> = args
                            .iter()
                            .map(|arg| match arg {
                                Arg::Pointer { idx } => args.as_ptr().add(*idx).cast_mut().cast(),
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
