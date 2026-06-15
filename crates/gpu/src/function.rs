use std::{
    collections::BTreeMap,
    rc::Rc,
    sync::{Arc, Mutex},
};

use bullet_compiler::{
    ir::NodeId,
    rewriterule,
    tensor::{
        DType, IRTrace, OpType, TType, TensorIR, TensorOp,
        operation::{
            BroadcastAcrossDimension, CABinary, CABinaryOp, Matmul, MatrixLayout, PadAcrossDimension,
            ReduceAcrossDimension, Reduction, ScalarConstant, SliceAcrossDimension,
        },
        transform::{
            IRTransform,
            eliminate::{EliminateCommonSubExpressions, EliminateUnusedOperations},
            modify::AddOperation,
            rewriterules::RewritePass,
        },
    },
};

use crate::{
    buffer::{Buffer, SyncOnDrop, SyncOnValue},
    kernel::KernelSrc,
    pointwise::transforms::{CodegenPointwise, FusePointwise, LowerPointwise},
    runtime::{Blas, Device, DeviceProps, Dialect, Dim3, GemmConfig, Gpu, Kernel, Module, Stream},
};

enum Inst<G: Gpu> {
    Malloc { idx: usize, ty: TType },
    Free { _idx: usize },
    Zero { idx: usize, ty: TType },
    LaunchKernel { func: Kernel<G>, args: Vec<usize>, gdim: Dim3, bdim: u32, smem: u32 },
    Matmul { cfg: Matmul, a: usize, b: usize, c: usize },
}

pub struct Function<G: Gpu> {
    device: Arc<Device<G>>,
    maps: BTreeMap<NodeId, (usize, bool, TType)>,
    insts: Box<[Inst<G>]>,
    blas: Option<Blas<G>>,
    allocated: bool,
    pointers: Mutex<Vec<G::DevicePtr>>,
}

impl<G: Gpu> Drop for Function<G> {
    fn drop(&mut self) {
        let _ = self.dealloc_preallocs();
    }
}

impl<G: Gpu> Function<G> {
    pub fn dealloc_preallocs(&mut self) -> Result<(), G::Error> {
        if !self.allocated {
            return Ok(());
        }

        let ptrs = self.pointers.lock().unwrap();
        for inst in &self.insts {
            if let &Inst::Malloc { idx, .. } = inst {
                unsafe { self.device.free(ptrs[idx])? };
            }
        }

        self.allocated = false;

        Ok(())
    }

    pub fn new(device: Arc<Device<G>>, mut ir: TensorIR) -> Result<Self, IRTrace> {
        let props = device.props().clone();
        ir.transform(RewritePass(MatmulToBroadcastMul))?;
        ir.transform(DuplicateScalarsAndIndexing)?;
        ir.transform(LowerPointwise(props.clone()))?;
        ir.transform(FusePointwise(props.clone()))?;
        ir.transform(RewritePass(ReduceToMatmul))?;
        ir.transform(EliminateCommonSubExpressions)?;
        ir.transform(LowerPointwise(props.clone()))?;
        ir.transform(CodegenPointwise(props.clone()))?;
        ir.transform(CodegenReduction(props.clone()))?;

        let mut maps = BTreeMap::new();
        let mut num_ptrs = 0;
        let mut insts = Vec::new();
        let mut requires_blas = false;

        let mut times_seen = BTreeMap::new();
        let mut indices = BTreeMap::new();

        let mut max_num_args = 0;

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
            if let Some(KernelSrc { name, source, arg_order, gdim, bdim, smem, requires_zero, .. }) =
                data.downcast().cloned()
            {
                let mut args = Vec::new();
                for (index, is_input) in arg_order {
                    let node_id = if is_input { op.inputs()[index] } else { op.outputs()[index] };
                    args.push(*indices.get(&node_id).unwrap());
                }

                for output in requires_zero {
                    let node_id = op.outputs()[output];
                    let ty = ir.get_node(node_id)?.ty();
                    insts.push(Inst::Zero { idx: *indices.get(&node_id).unwrap(), ty });
                }

                let func = Module::new(device.clone(), source.clone())
                    .map_err(|e| IRTrace::from(format!("{e:?}\n{source}")))?
                    .get_kernel(name)
                    .map_err(|e| IRTrace::from(format!("{e:?}\n{source}")))?;

                max_num_args = max_num_args.max(args.len());
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
                        insts.push(Inst::Free { _idx: idx });
                    }
                }
            }
        }

        let blas = requires_blas.then(|| Blas::new(device.clone()).unwrap());
        Ok(Self {
            device,
            maps,
            insts: insts.into_boxed_slice(),
            blas,
            allocated: false,
            pointers: Mutex::new(vec![G::DevicePtr::default(); num_ptrs]),
        })
    }

    pub fn prealloc(&mut self) -> Result<(), G::Error> {
        if self.allocated {
            return Ok(());
        }

        let mut ptrs = self.pointers.lock().unwrap();
        for inst in &self.insts {
            if let &Inst::Malloc { idx, ty } = inst {
                let bytes = ty.dtype().bytes() * ty.size().get();
                ptrs[idx] = self.device.malloc(bytes)?;
            }
        }

        self.allocated = true;

        Ok(())
    }

    pub fn execute(
        &self,
        stream: Arc<Stream<G>>,
        inputs: &BTreeMap<NodeId, Arc<Buffer<G>>>,
    ) -> Result<SyncOnValue<G, &Self>, G::Error> {
        let mut sync = SyncOnDrop::new(stream.clone());

        let mut ptrs = self.pointers.lock().unwrap();

        let mut mutmap = BTreeMap::new();

        for id in self.maps.keys() {
            if !inputs.contains_key(id) {
                return Err(format!("Input missing: {id:?}!").into());
            }
        }

        for (name, buf) in inputs {
            if let Some((idx, is_mut, ty)) = self.maps.get(name).cloned() {
                if buf.dtype() != ty.dtype() {
                    return Err("Mismatched dtypes!".to_string().into());
                }

                if buf.size() != ty.size().get() {
                    return Err("Mismatched sizes!".to_string().into());
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
        }

        if !self.allocated {
            return Err("Must allocate buffers for function before execution!".to_string().into());
        }

        unsafe {
            for inst in &self.insts {
                match inst {
                    &Inst::Zero { idx, ty } => {
                        let bytes = ty.size().get() * ty.dtype().bytes();
                        stream.memset(ptrs[idx], bytes, 0)?;
                    }
                    Inst::Malloc { .. } | Inst::Free { .. } => {}
                    Inst::LaunchKernel { func, args, gdim, bdim, smem } => {
                        let mut args: Vec<_> =
                            args.iter().map(|&arg| ptrs.as_ptr().add(arg).cast_mut().cast()).collect();

                        func.launch(&stream, *gdim, *bdim, &mut args, *smem)?;
                    }
                    &Inst::Matmul { cfg, a, b, c } => {
                        let handle = self.blas.as_ref().unwrap();
                        let config = GemmConfig {
                            row_mjr_a: !cfg.lhs.col_mjr,
                            row_mjr_b: !cfg.rhs.col_mjr,
                            m: cfg.lhs.rows.get().try_into().unwrap(),
                            n: cfg.rhs.cols.get().try_into().unwrap(),
                            k: cfg.lhs.cols.get().try_into().unwrap(),
                            alpha: 1.0,
                            beta: 0.0,
                        };

                        let batch = cfg.batch.get();
                        if batch == 1 {
                            handle.gemm(stream.as_ref(), config, ptrs[a], ptrs[b], ptrs[c])?;
                        } else {
                            handle.batched_gemm(stream.as_ref(), batch, config, ptrs[a], ptrs[b], ptrs[c])?;
                        }
                    }
                }
            }
        }

        Ok(SyncOnValue::new(sync, self))
    }
}

/// Separate out all `ScalarConst`s, as otherwise we end up
/// materialising them in kernel A and passing to kernel B,
/// rather than handling internally for each
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DuplicateScalarsAndIndexing;
impl IRTransform for DuplicateScalarsAndIndexing {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        for op in ir.operations() {
            for &input in op.inputs() {
                let grandparents = ir.get_op(ir.get_parent_op(input)?)?.inputs().to_vec();

                if let Some(&ScalarConstant(value, size)) = ir.parent_op(input)? {
                    let new_scalar = ir.add_scalar(value, size);
                    ir.ir_mut().replace_single_input(op.id(), new_scalar, input)?;
                } else if let Some(broadcast) = ir.parent_op::<BroadcastAcrossDimension>(input)? {
                    let broadcast = ir.add_op(grandparents, Ok::<_, IRTrace>(*broadcast))?[0];
                    ir.ir_mut().replace_single_input(op.id(), broadcast, input)?;
                } else if let Some(slice) = ir.parent_op::<SliceAcrossDimension>(input)? {
                    let slice = ir.add_op(grandparents, Ok::<_, IRTrace>(*slice))?[0];
                    ir.ir_mut().replace_single_input(op.id(), slice, input)?;
                } else if let Some(pad) = ir.parent_op::<PadAcrossDimension>(input)? {
                    let pad = ir.add_op(grandparents, Ok::<_, IRTrace>(*pad))?[0];
                    ir.ir_mut().replace_single_input(op.id(), pad, input)?;
                }
            }
        }

        ir.transform(EliminateUnusedOperations)
    }
}

// I don't want to write reduction kernels right now so scam it with matmul
rewriterule! {
    rulename ReduceToMatmul on ir
    rewrites op (output = [ReduceAcrossDimension] (input))
    {
        if output.dtype() == DType::F32 && output.reduction() == Reduction::Sum {
            let input = input.id();

            let (new_scalar, new_op) = if output.inner().get() == 1 {
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

// Rewrite Mx1 @ 1xN to broadcast and pointwise multiplication
rewriterule! {
    rulename MatmulToBroadcastMul on ir
    rewrites op (output = [Matmul] (lhs) (rhs))
    {
        if output.lhs.cols.get() == 1 {
            let m = output.lhs.rows;
            let n = output.rhs.cols;

            let lhs = lhs.id();
            let rhs = rhs.id();
            let lhs = ir.add_broadcast(lhs, [m], 0, n)?;
            let rhs = ir.add_broadcast(rhs, [n, 1.into()], 1, m)?;
            let ty = ir.get_node(lhs)?.ty();
            let new_op = CABinaryOp::new(ty, CABinary::Mul);

            ir.replace_operation(op.id(), [lhs, rhs], new_op)?;
            return Ok(true);
        }
    }
}

static REDUCTION_SRC_CUDA: &str = "
extern \"C\" __global__ void reduce_kernel(const float* input, float* output) {
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < OUTER) {
        float reduction = input[INNER * tid];

        for (int i = 1; i < INNER; i++) {
            reduction = FUNC(reduction, input[INNER * tid + i]);
        }

        output[tid] = reduction;
    }
}";

static REDUCTION_SRC_MSL: &str = "
#include <metal_stdlib>
using namespace metal;
kernel void reduce_kernel(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if (tid < (OUTER)) {
        float reduction = input[INNER * tid];

        for (int i = 1; i < INNER; i++) {
            reduction = FUNC(reduction, input[INNER * tid + i]);
        }

        output[tid] = reduction;
    }
}";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodegenReduction(pub DeviceProps);

impl IRTransform for CodegenReduction {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        for op in ir.operations() {
            if let Some(reduction) = op.data().downcast::<ReduceAcrossDimension>()
                && reduction.reduction() != Reduction::Sum
                && reduction.inner() == 1.into()
            {
                let outer = reduction.outer().get();
                let dimen = reduction.dimen().get();

                let src = (match self.0.dialect() {
                    Dialect::CudaHip => REDUCTION_SRC_CUDA,
                    Dialect::Msl => REDUCTION_SRC_MSL,
                })
                .replace(
                    "FUNC",
                    match reduction.reduction() {
                        Reduction::Max => "max",
                        Reduction::Min => "min",
                        _ => unimplemented!(),
                    },
                )
                .replace("INNER", &dimen.to_string())
                .replace("OUTER", &outer.to_string());

                let new = unsafe {
                    KernelSrc::new(
                        reduction.inputs(),
                        reduction.outputs(),
                        "reduce_kernel".to_string(),
                        src,
                        vec![(0, true), (0, false)],
                        Default::default(),
                        Dim3 { x: outer.div_ceil(256).try_into().unwrap(), y: 1, z: 1 },
                        256,
                        0,
                    )
                };

                ir.replace_op(op.id(), AddOperation::new(op.inputs(), Ok::<_, IRTrace>(TensorOp(Rc::new(new)))))?;
            }
        }

        Ok(())
    }
}
