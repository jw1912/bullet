use std::{collections::HashMap, rc::Rc, sync::Arc};

use bullet_compiler::{
    ir::NodeId,
    rewriterule,
    tensor::{
        DType, IRTrace, Size, TType, TensorIR,
        operation::{Matmul, MatrixLayout, ReduceAcrossDimension, Reduction},
        transform::rewriterules::RewritePass,
    },
};

use crate::{
    buffer::{Buffer, SyncOnDrop, SyncOnValue},
    kernel::KernelSrc,
    pointwise::LowerPointwise,
    runtime::{Blas, Device, Dim3, GemmConfig, Gpu, Kernel, Module, Stream},
};

#[derive(Debug)]
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
    Matmul {
        cfg: Matmul,
        a: usize,
        b: usize,
        c: usize,
    },
}

pub struct Function<G: Gpu> {
    maps: HashMap<NodeId, (usize, bool, TType)>,
    insts: Box<[Inst<G>]>,
    num_ptrs: usize,
    blas: Option<Blas<G>>,
}

impl<G: Gpu> Function<G> {
    pub fn new(device: Arc<Device<G>>, mut ir: TensorIR) -> Result<Self, IRTrace> {
        ir.transform(RewritePass(ReduceToMatmul))?;
        ir.transform(LowerPointwise)?;

        let mut maps = HashMap::new();
        let mut num_ptrs = 0;
        let mut insts = Vec::new();
        let mut requires_blas = false;

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

                let func = Module::new(device.clone(), source.clone())
                    .map_err(|e| IRTrace::from(format!("{e:?}\n{source}")))?
                    .get_kernel("kernel")
                    .map_err(|e| IRTrace::from(format!("{e:?}\n{source}")))?;

                insts.push(Inst::LaunchKernel { func, args, gdim, bdim, smem });
            } else if let Some(cfg) = data.downcast::<Matmul>().cloned() {
                if cfg.dtype != DType::F32 {
                    return Err("Unsupported matmul dtype!".into());
                }

                let [a, b] = op.inputs()[..] else { return Err("Invalid inputs!".into()) };
                let [c] = op.outputs()[..] else { return Err("Invalid inputs!".into()) };

                requires_blas = true;
                insts.push(Inst::Matmul {
                    cfg,
                    a: *indices.get(&a).unwrap(),
                    b: *indices.get(&b).unwrap(),
                    c: *indices.get(&c).unwrap(),
                })
            } else if !data.is_input() {
                return Err(format!("Unsupported operation: {data:?}").into());
            }

            // free buffers that see no more usage
            for &input in op.inputs() {
                if !ir.is_input(input)? && !ir.is_output(input) {
                    let times_seen = times_seen.get_mut(&input).unwrap();
                    *times_seen += 1;

                    if ir.get_node(input)?.children() == *times_seen {
                        let idx = *indices.get(&input).unwrap();
                        insts.push(Inst::Free { idx });
                    }
                }
            }
        }

        let blas = requires_blas.then(|| Blas::new(device).unwrap());
        Ok(Self { maps, insts: insts.into_boxed_slice(), num_ptrs, blas })
    }

    pub fn execute(
        &self,
        stream: Arc<Stream<G>>,
        inputs: &HashMap<NodeId, Arc<Buffer<G>>>,
    ) -> Result<SyncOnValue<G, &Self>, G::Error> {
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
            for inst in &self.insts {
                match inst {
                    &Inst::Malloc { idx, ty } => {
                        let bytes = ty.size().evaluate(var) * ty.dtype().bytes();
                        ptrs[idx] = stream.malloc(bytes)?;
                    }
                    &Inst::Free { idx } => {
                        stream.free(ptrs[idx])?;
                    }
                    &Inst::Zero { idx, ty } => {
                        let bytes = ty.size().evaluate(var) * ty.dtype().bytes();
                        stream.memset(ptrs[idx], bytes, 0)?;
                    }
                    Inst::LaunchKernel { func, args, gdim, bdim, smem } => {
                        sizes.clear();
                        let mut args: Vec<_> = args
                            .iter()
                            .map(|arg| match arg {
                                Arg::Pointer { idx } => ptrs.as_ptr().add(*idx).cast_mut().cast(),
                                Arg::Size(size) => {
                                    sizes.push(size.evaluate(var) as i32);
                                    (sizes.last().unwrap() as *const i32).cast_mut().cast()
                                }
                            })
                            .collect();

                        func.launch(&stream, gdim(var), bdim(var), args.as_mut_ptr(), smem(var))?;
                    }
                    &Inst::Matmul { cfg, a, b, c } => {
                        let handle = self.blas.as_ref().unwrap();
                        let config = GemmConfig {
                            row_mjr_a: !cfg.lhs.col_mjr,
                            row_mjr_b: !cfg.rhs.col_mjr,
                            m: cfg.lhs.rows.evaluate(var).try_into().unwrap(),
                            n: cfg.rhs.cols.evaluate(var).try_into().unwrap(),
                            k: cfg.lhs.cols.evaluate(var).try_into().unwrap(),
                            alpha: 1.0,
                            beta: 0.0,
                        };

                        if let Some(1) = cfg.batch.evaluate_constant() {
                            handle.gemm(stream.as_ref(), config, ptrs[a], ptrs[b], ptrs[c])?;
                        } else {
                            let batch = cfg.batch.evaluate(var);
                            handle.batched_gemm(stream.as_ref(), batch, config, ptrs[a], ptrs[b], ptrs[c])?;
                        }
                    }
                }
            }
        }

        Ok(SyncOnValue::new(sync, self))
    }
}

rewriterule! {
    rulename ReduceToMatmul on ir
    rewrites op (output = [ReduceAcrossDimension] (input))
    {
        if output.dtype() == DType::F32 && output.reduction() == Reduction::Sum {
            let input = input.id();

            let (new_scalar, new_op) = if let Some(1) = output.inner().evaluate_constant() {
                let new_scalar = ir.add_scalar(1.0, output.dimen());
                let lhs = MatrixLayout { rows: 1.into(), cols: output.dimen(), col_mjr: true };
                let rhs = MatrixLayout { rows: output.dimen(), cols: output.outer(), col_mjr: true };
                (new_scalar, Matmul::new(DType::F32, 1, lhs, rhs)?)
            } else {
                let new_scalar = ir.add_scalar(1.0, output.outer() * output.dimen());
                let lhs = MatrixLayout { rows: 1.into(), cols: output.dimen(), col_mjr: true };
                let rhs = MatrixLayout { rows: output.dimen(), cols: output.inner(), col_mjr: false };
                (new_scalar, Matmul::new(DType::F32, output.outer(), lhs, rhs)?)
            };

            ir.replace_operation(op.id(), [new_scalar, input], new_op)?;
            return Ok(true);
        }
    }
}
